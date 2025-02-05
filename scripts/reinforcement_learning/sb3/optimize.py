# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--algo",
    type=str,
    default="ppo",
    help="Name of the algorithm.",
    choices=["ppo", "sac", "tqc"],
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--no-info",
    action="store_true",
    default=False,
    help="Fastest and incorrect training but no statistics.",
)
# parser.add_argument("--monitor", action="store_true", default=False, help="Enable VecMonitor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
import time
from datetime import datetime

import flax
import optax
import sbx

# from stable_baselines3 import PPO
from isaaclab_rl.sb3 import ClipActionWrapper, RescaleActionWrapper, Sb3VecEnvWrapper, process_sb3_cfg
from stable_baselines3.common.callbacks import BaseCallback

# from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


class TimeoutCallback(BaseCallback):
    def __init__(self, timeout: int = 60, verbose: int = 0):
        super().__init__(verbose)
        # Timeout in second
        self.timeout = timeout
        self.start_time = None

    def _on_step(self) -> bool:
        if not self.start_time:
            self.start_time = time.time()

        return (time.time() - self.start_time) > self.timeout


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    # if args_cli.max_iterations is not None:
    #     agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3_optim", args_cli.algo, args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    # policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=True, keep_info=not args_cli.no_info)

    if args_cli.algo != "ppo":
        env = RescaleActionWrapper(env, percent=3)
    # else:
    #     env = ClipActionWrapper(env, percent=3)
    #     # env = RescaleActionWrapper(env, percent=3)
    # env = ClipActionWrapper(env, percent=3.0)
    # env = RescaleActionWrapper(env, percent=3.0)

    print(f"Action space: {env.action_space}")

    # import ipdb
    # ipdb.set_trace()

    print("Normalizing input")
    env = VecNormalize(
        env,
        training=True,
        norm_obs=True,
        # norm_reward="normalize_value" in agent_cfg
        # and agent_cfg.pop("normalize_value"),
        # clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        # gamma=agent_cfg["gamma"],
        # clip_reward=np.inf,
    )

    simba_hyperparams = dict(
        policy="SimbaPolicy",
        # batch_size=256,
        buffer_size=500_000,
        # learning_rate=3e-4,
        policy_kwargs={
            "optimizer_class": optax.adamw,
            "activation_fn": flax.linen.elu,
            "net_arch": {"pi": [128, 128], "qf": [256, 256]},
            "n_critics": 2,
        },
        # learning_starts=10_000,
        learning_starts=1_000,
        # normalize={"norm_obs": True, "norm_reward": False},
        # param_resets=[int(i * 1e7) for i in range(1, 10)],
    )
    # ppo_hyperparams = dict(
    #     policy="MlpPolicy",
    #     policy_kwargs=dict(
    #         activation_fn=flax.linen.elu,
    #         # net_arch=[512, 256, 128],
    #         net_arch=[128, 128, 128],
    #         layer_norm=True,
    #     ),
    #     learning_starts=1_000,
    # )
    # hyperparams = ppo_hyperparams
    hyperparams = simba_hyperparams

    log_interval = 100
    if args_cli.algo == "tqc":
        n_timesteps = int(3e7)
        agent = sbx.SAC(
            env=env,
            train_freq=5,
            # learning_rate=7e-4,
            gamma=0.985,
            batch_size=512,
            # gradient_steps=min(env.num_envs, 256),
            gradient_steps=min(env.num_envs, 512),
            policy_delay=10,
            verbose=1,
            # ent_coef=0.001,
            ent_coef="auto_0.01",
            **hyperparams,
        )

    print(f"{env.num_envs=}")
    # train the agent
    agent.learn(
        total_timesteps=n_timesteps,
        log_interval=log_interval,
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
