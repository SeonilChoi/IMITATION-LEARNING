# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Bdx-Ase-Direct-v0",
    entry_point=f"{__name__}.bdx_ase_env:BdxAseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bdx_ase_env_cfg:BdxAseEnvCfg",
        "skrl_ase_cfg_entry_point": f"{agents.__name__}:skrl_ase_cfg.yaml",
    },
)

gym.register(
    id="Template-Bdx-Amp-Direct-v0",
    entry_point=f"{__name__}.bdx_amp_env:BdxAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bdx_amp_env_cfg:BdxAmpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)
