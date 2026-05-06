from __future__ import annotations

import copy
import dataclasses
import itertools
from typing import Any, Mapping, Optional, Tuple, Type, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch.amp import AMP, AMP_CFG
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.runner.torch import Runner


AMP_CFG_FIELDS = {field.name for field in dataclasses.fields(AMP_CFG)}
ASE_CUSTOM_DEFAULT_CONFIG = {
    "mi_reward_scale": 0.1,
    "encoder_loss_scale": 1.0,
    "latent_dim": 64,
    "discriminator_reward_scale": 1.0,
}

ASE_DEFAULT_CONFIG = dataclasses.asdict(AMP_CFG())
ASE_DEFAULT_CONFIG.update(
    ASE_CUSTOM_DEFAULT_CONFIG
)


def _normalize_amp_cfg_keys(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Map legacy skrl AMP config keys to the skrl 2.x names."""
    normalized_cfg = copy.deepcopy(dict(cfg))
    aliases = {
        "amp_state_preprocessor": "amp_observation_preprocessor",
        "amp_state_preprocessor_kwargs": "amp_observation_preprocessor_kwargs",
        "task_reward_weight": "task_reward_scale",
        "style_reward_weight": "style_reward_scale",
        "lambda": "gae_lambda",
    }
    for old_key, new_key in aliases.items():
        if old_key in normalized_cfg and new_key not in normalized_cfg:
            normalized_cfg[new_key] = normalized_cfg[old_key]
        normalized_cfg.pop(old_key, None)

    if "clip_predicted_values" in normalized_cfg:
        value_clip = normalized_cfg.get("value_clip", 0.2)
        normalized_cfg["value_clip"] = value_clip if normalized_cfg.pop("clip_predicted_values") else 0.0

    normalized_cfg.pop("class", None)
    return normalized_cfg


class ASE(AMP):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        state_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        amp_observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        motion_dataset: Optional[Memory] = None,
        reply_buffer: Optional[Memory] = None,
        collect_reference_motions=None,
    ) -> None:
        _cfg = copy.deepcopy(ASE_DEFAULT_CONFIG)
        _cfg.update(_normalize_amp_cfg_keys(cfg if cfg is not None else {}))

        self._mi_reward_scale = _cfg.pop("mi_reward_scale")
        self._encoder_loss_scale = _cfg.pop("encoder_loss_scale")
        self._latent_dim = _cfg.pop("latent_dim")
        self._discriminator_reward_scale = _cfg.pop("discriminator_reward_scale")
        amp_cfg = {key: value for key, value in _cfg.items() if key in AMP_CFG_FIELDS}

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=amp_cfg,
            amp_observation_space=amp_observation_space,
            motion_dataset=motion_dataset,
            reply_buffer=reply_buffer,
            collect_reference_motions=collect_reference_motions,
        )

        self.encoder = self.models.get("encoder", None)
        if self.encoder is None:
            raise KeyError("ASE agent requires an 'encoder' model")

        self.checkpoint_modules["encoder"] = self.encoder

        if config.torch.is_distributed:
            self.encoder.broadcast_parameters()

        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.policy.parameters(),
                self.value.parameters(),
                self.discriminator.parameters(),
                self.encoder.parameters(),
            ),
            lr=self.cfg.learning_rate[0],
        )
        self.scheduler = self.cfg.learning_rate_scheduler[0]
        if self.scheduler is not None:
            self.scheduler = self.cfg.learning_rate_scheduler[0](
                self.optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
            )
        self.checkpoint_modules["optimizer"] = self.optimizer

    def _compute_style_reward(self, amp_observations: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            amp_logits, _ = self.discriminator.act(
                {"observations": self._amp_observation_preprocessor(amp_observations)}, role="discriminator"
            )
            style_reward = -torch.log(
                torch.maximum(1 - 1 / (1 + torch.exp(-amp_logits)), torch.tensor(0.0001, device=self.device))
            )
            style_reward *= self._discriminator_reward_scale
            return style_reward.view(rewards.shape)

    def _compute_info_reward(
        self, amp_observations: torch.Tensor, latents: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        flat_amp_observations = amp_observations.view(-1, amp_observations.shape[-1])
        flat_latents = latents.view(-1, latents.shape[-1])
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            encoder_output, _ = self.encoder.act(
                {"observations": self._amp_observation_preprocessor(flat_amp_observations)}, role="encoder"
            )
            encoder_output = F.normalize(encoder_output, dim=-1)
            normalized_latents = F.normalize(flat_latents, dim=-1)
            info_reward = self._mi_reward_scale * torch.sum(encoder_output * normalized_latents, dim=-1, keepdim=True)
            return info_reward.view(rewards.shape)

    def update(self, *, timestep: int, timesteps: int) -> None:
        def compute_gae(
            rewards: torch.Tensor,
            terminated: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_terminated = terminated.logical_not()
            memory_size = rewards.shape[0]

            for i in reversed(range(memory_size)):
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * (next_values[i] + lambda_coefficient * not_terminated[i] * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        self.motion_dataset.add_samples(observations=self.collect_reference_motions(self.cfg.amp_batch_size))

        rewards = self.memory.get_tensor_by_name("rewards")
        observations = self.memory.get_tensor_by_name("observations")
        states = self.memory.tensors.get("states")
        amp_observations = self.memory.get_tensor_by_name("amp_observations")
        latent_source = states if states is not None else observations
        latents = latent_source[..., -self._latent_dim :]

        style_reward = self._compute_style_reward(amp_observations, rewards)
        info_reward = self._compute_info_reward(amp_observations, latents, rewards)
        combined_rewards = self.cfg.task_reward_scale * rewards + self.cfg.style_reward_scale * style_reward + info_reward

        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")
        returns, advantages = compute_gae(
            rewards=combined_rewards,
            terminated=self.memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=next_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self.cfg.mini_batches)
        sampled_motion_batches = self.motion_dataset.sample(
            names=["observations"],
            batch_size=self.memory.memory_size * self.memory.num_envs,
            mini_batches=self.cfg.mini_batches,
        )
        if len(self.reply_buffer):
            sampled_replay_batches = self.reply_buffer.sample(
                names=["observations"],
                batch_size=self.memory.memory_size * self.memory.num_envs,
                mini_batches=self.cfg.mini_batches,
            )
        else:
            sampled_replay_batches = [
                [batches[self._tensors_names.index("amp_observations")]] for batches in sampled_batches
            ]

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_discriminator_loss = 0.0
        cumulative_encoder_loss = 0.0

        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []
            for batch_index, (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_amp_observations,
            ) in enumerate(sampled_batches):
                sampled_latent_source = sampled_states if sampled_states is not None else sampled_observations
                sampled_latents = F.normalize(sampled_latent_source[:, -self._latent_dim :], dim=-1)

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": self._state_preprocessor(sampled_states, train=not epoch)
                        if sampled_states is not None
                        else None,
                    }

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _ = self.value.act(inputs, role="value")
                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    if self.cfg.discriminator_batch_size > 0:
                        sampled_amp_observations = self._amp_observation_preprocessor(
                            sampled_amp_observations[0 : self.cfg.discriminator_batch_size], train=True
                        )
                        sampled_amp_replay_observations = self._amp_observation_preprocessor(
                            sampled_replay_batches[batch_index][0][0 : self.cfg.discriminator_batch_size], train=True
                        )
                        sampled_amp_motion_observations = self._amp_observation_preprocessor(
                            sampled_motion_batches[batch_index][0][0 : self.cfg.discriminator_batch_size], train=True
                        )
                        sampled_latents = sampled_latents[0 : self.cfg.discriminator_batch_size]
                    else:
                        sampled_amp_observations = self._amp_observation_preprocessor(
                            sampled_amp_observations, train=True
                        )
                        sampled_amp_replay_observations = self._amp_observation_preprocessor(
                            sampled_replay_batches[batch_index][0], train=True
                        )
                        sampled_amp_motion_observations = self._amp_observation_preprocessor(
                            sampled_motion_batches[batch_index][0], train=True
                        )

                    sampled_amp_motion_observations.requires_grad_(True)
                    amp_logits, _ = self.discriminator.act(
                        {"observations": sampled_amp_observations}, role="discriminator"
                    )
                    amp_replay_logits, _ = self.discriminator.act(
                        {"observations": sampled_amp_replay_observations}, role="discriminator"
                    )
                    amp_motion_logits, _ = self.discriminator.act(
                        {"observations": sampled_amp_motion_observations}, role="discriminator"
                    )

                    amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)
                    discriminator_loss = 0.5 * (
                        nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits))
                        + nn.BCEWithLogitsLoss()(amp_motion_logits, torch.ones_like(amp_motion_logits))
                    )

                    if self.cfg.discriminator_logit_regularization_scale:
                        logit_weights = torch.flatten(list(self.discriminator.modules())[-1].weight)
                        discriminator_loss += self.cfg.discriminator_logit_regularization_scale * torch.sum(
                            torch.square(logit_weights)
                        )

                    if self.cfg.discriminator_gradient_penalty_scale:
                        amp_motion_gradient = torch.autograd.grad(
                            amp_motion_logits,
                            sampled_amp_motion_observations,
                            grad_outputs=torch.ones_like(amp_motion_logits),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )
                        gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                        discriminator_loss += self.cfg.discriminator_gradient_penalty_scale * gradient_penalty

                    if self.cfg.discriminator_weight_decay_scale:
                        weights = [
                            torch.flatten(module.weight)
                            for module in self.discriminator.modules()
                            if isinstance(module, torch.nn.Linear)
                        ]
                        weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                        discriminator_loss += self.cfg.discriminator_weight_decay_scale * weight_decay

                    discriminator_loss *= self.cfg.discriminator_loss_scale

                    encoder_output, _ = self.encoder.act(
                        {"observations": sampled_amp_observations}, role="encoder"
                    )
                    encoder_output = F.normalize(encoder_output, dim=-1)
                    encoder_loss = -self._encoder_loss_scale * torch.mean(
                        torch.sum(encoder_output * sampled_latents, dim=-1)
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + discriminator_loss + encoder_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    self.value.reduce_parameters()
                    self.discriminator.reduce_parameters()
                    self.encoder.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.policy.parameters(),
                            self.value.parameters(),
                            self.discriminator.parameters(),
                            self.encoder.parameters(),
                        ),
                        self.cfg.grad_norm_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_discriminator_loss += discriminator_loss.item()
                cumulative_encoder_loss += encoder_loss.item()

            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        self.reply_buffer.add_samples(observations=amp_observations.view(-1, amp_observations.shape[-1]))

        denom = self.cfg.learning_epochs * self.cfg.mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denom)
        self.track_data("Loss / Value loss", cumulative_value_loss / denom)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / denom)
        self.track_data("Loss / Discriminator loss", cumulative_discriminator_loss / denom)
        self.track_data("Loss / Encoder loss", cumulative_encoder_loss / denom)
        self.track_data("Reward / Style reward (mean)", style_reward.mean().item())
        self.track_data("Reward / MI reward (mean)", info_reward.mean().item())
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])


class ASERunner(Runner):
    def _component(self, name: str) -> Type:
        if name.lower() in ["ase", "ase_cfg", "ase_default_config"]:
            return ASE_DEFAULT_CONFIG if "cfg" in name.lower() or "default_config" in name.lower() else ASE
        return super()._component(name)

    def _generate_models(self, env, cfg: Mapping[str, Any]):
        device = env.device
        agent_id = "agent"
        observation_space = env.observation_space
        action_space = env.action_space
        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if agent_class != "ase":
            return super()._generate_models(env, cfg)

        amp_observation_space = env.amp_observation_space
        latent_dim = cfg["agent"]["latent_dim"]
        latent_action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32
        )
        models_cfg = copy.deepcopy(cfg.get("models"))
        if not models_cfg:
            raise ValueError("No 'models' are defined in cfg")
        separate = models_cfg.pop("separate", True)
        if not separate:
            raise ValueError("ASERunner only supports separate models")

        models = {agent_id: {}}
        for role in models_cfg:
            model_class_name = models_cfg[role].pop("class")
            model_class = self._component(model_class_name)
            role_observation_space = amp_observation_space if role in ["discriminator", "encoder"] else observation_space
            role_action_space = latent_action_space if role == "encoder" else action_space

            source = model_class(
                observation_space=role_observation_space,
                action_space=role_action_space,
                device=device,
                **self._process_cfg(models_cfg[role]),
                return_source=True,
            )
            print("==================================================")
            print(f"Model (role): {role}")
            print("==================================================\n")
            print(source)
            print("--------------------------------------------------")

            models[agent_id][role] = model_class(
                observation_space=role_observation_space,
                action_space=role_action_space,
                device=device,
                **self._process_cfg(models_cfg[role]),
            )

        for role, model in models[agent_id].items():
            model.init_state_dict(role=role)
        return models

    def _generate_agent(self, env, cfg: Mapping[str, Any], models):
        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if agent_class != "ase":
            return super()._generate_agent(env, cfg, models)

        device = env.device
        num_envs = env.num_envs
        observation_space = env.observation_space
        state_space = getattr(env, "state_space", observation_space)
        action_space = env.action_space
        amp_observation_space = env.amp_observation_space

        memory_cfg = copy.deepcopy(cfg["memory"])
        memory_class = self._component(memory_cfg.pop("class"))
        if memory_cfg["memory_size"] < 0:
            memory_cfg["memory_size"] = cfg["agent"]["rollouts"]
        memory = memory_class(num_envs=num_envs, device=device, **self._process_cfg(memory_cfg))

        motion_dataset = None
        if cfg.get("motion_dataset"):
            motion_dataset_cfg = copy.deepcopy(cfg["motion_dataset"])
            motion_dataset_class = self._component(motion_dataset_cfg.pop("class"))
            motion_dataset = motion_dataset_class(device=device, **self._process_cfg(motion_dataset_cfg))

        reply_buffer = None
        if cfg.get("reply_buffer"):
            reply_buffer_cfg = copy.deepcopy(cfg["reply_buffer"])
            reply_buffer_class = self._component(reply_buffer_cfg.pop("class"))
            reply_buffer = reply_buffer_class(device=device, **self._process_cfg(reply_buffer_cfg))

        agent_cfg = self._component("ase_default_config").copy()
        agent_cfg.update(self._process_cfg(_normalize_amp_cfg_keys(cfg["agent"])))
        agent_cfg.setdefault("observation_preprocessor_kwargs", {}).update(
            {"size": observation_space, "device": device}
        )
        agent_cfg.setdefault("state_preprocessor_kwargs", {}).update({"size": state_space, "device": device})
        agent_cfg.setdefault("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
        agent_cfg.setdefault("amp_observation_preprocessor_kwargs", {}).update(
            {"size": amp_observation_space, "device": device}
        )

        return ASE(
            models=models["agent"],
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=agent_cfg,
            amp_observation_space=amp_observation_space,
            motion_dataset=motion_dataset,
            reply_buffer=reply_buffer,
            collect_reference_motions=lambda num_samples: env.collect_reference_motions(num_samples),
        )
