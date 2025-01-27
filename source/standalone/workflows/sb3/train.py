# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 512


"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the algorithm.", choices=["ppo", "sac", "tqc"])
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--fast", action="store_true", default=False, help="Faster correct training but not extras logged.")
parser.add_argument(
    "--no-info", action="store_true", default=False, help="Fastest and incorrect training but no statistics."
)
# parser.add_argument("--monitor", action="store_true", default=False, help="Enable VecMonitor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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
import torch
from datetime import datetime

import flax
import optax
import sbx

# from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.sb3 import RescaleActionWrapper, Sb3VecEnvWrapper, process_sb3_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
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
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    # policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(
        env,
        fast_variant=args_cli.fast,
        keep_info=not args_cli.no_info,
    )

    if args_cli.algo != "ppo":
        env = RescaleActionWrapper(env, percent=2.5)
    # import ipdb
    # ipdb.set_trace()

    if "normalize_input" in agent_cfg:
        print("Normalizing input")
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    simba_hyperparams = dict(
        # batch_size=256,
        # buffer_size=100_000,
        # learning_rate=3e-4,
        policy_kwargs={
            "optimizer_class": optax.adamw,
            "net_arch": {"pi": [128], "qf": [256, 256]},
            "n_critics": 2,
        },
        learning_starts=10_000,
        # normalize={"norm_obs": True, "norm_reward": False},
        # resets=[50000, 75000],
    )
    if args_cli.algo == "tqc":
        agent = sbx.TQC(
            "SimbaPolicy",
            env,
            train_freq=5,
            learning_rate=1e-3,
            batch_size=256,
            gradient_steps=min(env.num_envs, 256),
            policy_delay=10,
            verbose=1,
            ent_coef=0.01,
            **simba_hyperparams,
        )
    elif args_cli.algo == "ppo":
        n_timesteps = int(3e7)

        hyperparams = dict(
            policy_kwargs=dict(
                activation_fn=flax.linen.elu,
                # net_arch=[512, 256, 128],
                net_arch=[128, 128, 128],
            )
        )

        # import torch
        # import stable_baselines3 as sb3
        # import warnings
        # warnings.simplefilter()
        # hyperparams = dict(
        #     policy_kwargs=dict(
        #         activation_fn=torch.nn.ELU,
        #         net_arch=[128, 128, 128],
        #     )
        # )
        # agent = sb3.PPO("MlpPolicy", env, verbose=1, **agent_cfg, **hyperparams)
        agent = sbx.PPO("MlpPolicy", env, verbose=1, **agent_cfg, **hyperparams)
    elif args_cli.algo == "sac":
        agent = sbx.SAC(
            "MlpPolicy",
            env,
            train_freq=5,
            gradient_steps=min(env.num_envs, 256),
            policy_delay=10,
            verbose=1,
        )
    # configure the logger
    # new_logger = configure(log_dir, ["stdout", "tensorboard"])
    # agent.set_logger(new_logger)

    print(f"{env.num_envs=}")
    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=log_dir, name_prefix="model", verbose=2)
    # checkpoint_callback = None
    # train the agent
    try:
        agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback, progress_bar=True, log_interval=20)
    except KeyboardInterrupt:
        pass
    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
