# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1024 --fast

"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from copy import deepcopy
from pathlib import Path
from pprint import pprint

from isaaclab.app import AppLauncher


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--algo", type=str, default="ppo", help="Name of the algorithm.", choices=["ppo", "sac", "tqc", "ppo_sb3"]
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log-interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--fast", action="store_true", default=False, help="Faster correct training but not extras logged.")
parser.add_argument(
    "--no-info", action="store_true", default=False, help="Fastest and incorrect training but no statistics."
)
parser.add_argument(
    "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
)
parser.add_argument("-name", "--study-name", help="Study name when loading Optuna results", type=str)
parser.add_argument("-id", "--trial-id", help="Trial id to load, otherwise loading best trial", type=int)
parser.add_argument(
    "-params",
    "--hyperparams",
    type=str,
    nargs="+",
    action=StoreDict,
    help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
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


def cleanup_pbar(*args):
    # Cleanup pbar
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# Disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
import torch
from datetime import datetime

import jax
import optax
import sbx

# from stable_baselines3 import PPO
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.sb3 import LogEveryNTimesteps, RescaleActionWrapper, Sb3VecEnvWrapper, elu, load_trial, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

ppo_defaults = dict(
    n_steps=25,
    batch_size=6400,  # for 1024 envs, to have 4 minibatches
    gae_lambda=0.95,
    n_epochs=5,
    ent_coef=0.01,
    learning_rate=1e-3,
    clip_range=0.2,
    vf_coef=1.0,
    max_grad_norm=1.0,
    policy="MlpPolicy",
    policy_kwargs=dict(
        activation_fn=elu,
        net_arch=[512, 256, 128],
        # net_arch=[128, 128, 128],
        # log_std_init=-2.5,
    ),
)

ppo_simba = dict(
    policy="SimbaPolicy",
    policy_kwargs=dict(
        activation_fn=elu,
        # net_arch=[512, 256, 128],
        net_arch=[128, 128],
    ),
)

ppo_sb3 = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(
        activation_fn=torch.nn.ELU,
        net_arch=[128, 128, 128],
        # log_std_init=-2.0,
        # use_expln=True,
        # squash_output=True,
    ),
    # use_sde=True,
    # sde_sample_freq=8,
    # TODO: use AdamW too
)
ppo_sb3_defaults = deepcopy(ppo_defaults)
ppo_sb3_defaults.update(ppo_sb3)

simba_hyperparams = dict(
    policy="SimbaPolicy",
    buffer_size=800_000,
    policy_kwargs={
        "optimizer_class": optax.adamw,
        # "optimizer_kwargs": {"eps": 1e-5},
        "activation_fn": elu,
        "net_arch": {"pi": [128, 128], "qf": [256, 256]},
        # "net_arch": [128, 128, 128],
        "n_critics": 2,
    },
    learning_starts=1_000,
    # param_resets=[int(i * 1e7) for i in range(1, 10)],
    train_freq=5,
    # learning_rate=7e-4,
    gamma=0.985,
    batch_size=512,
    gradient_steps=512,
    policy_delay=10,
    # ent_coef=0.001,
    ent_coef="auto_0.01",
    # target_entropy=-10.0,
    # tau=0.008,
    # top_quantiles_to_drop_per_net=5,
)

# Optimized with TQC on A1 flat for 2048 envs
optimized_tqc_hyperparams = dict(
    policy="MlpPolicy",
    buffer_size=800_000,
    policy_kwargs={
        "optimizer_class": optax.adamw,
        "activation_fn": elu,
        "net_arch": [512, 256, 128],
        # "net_arch": {"pi": [128, 128, 128], "qf": [512, 256, 128]},
        "n_critics": 2,
        "layer_norm": True,
    },
    learning_starts=1_000,
    # param_resets=[int(i * 1e7) for i in range(1, 10)],
    train_freq=4,
    # learning_rate=0.000375,
    learning_rate=4e-4,
    qf_learning_rate=7e-4,
    gamma=0.981,
    batch_size=256,
    gradient_steps=650,
    policy_delay=30,
    ent_coef="auto_0.00631",
)
# Also working for TQC, with 1024 envs:
# train_freq:4 gradient_steps:60 policy_delay:5 batch_size:1024
# train_freq:4 gradient_steps:30 policy_delay:5 batch_size:2048
# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if not agent_cfg:
        print("Loading SB3 default")
        agent_cfg = {
            "n_timesteps": 5e7,
            # "n_timesteps": 5e7,
            "normalize_input": True,
            "normalize_value": False,
            "clip_obs": 10.0,
            "seed": 42,
            "gamma": 0.99,
        }

        default_hyperparams = {
            "ppo_sb3": ppo_sb3_defaults,
            "ppo": ppo_defaults,
            "tqc": optimized_tqc_hyperparams,
            "sac": optimized_tqc_hyperparams,
        }[args_cli.algo]

        agent_cfg.update(default_hyperparams)

        pprint(agent_cfg)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.algo, args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # read configurations about the agent-training
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
    env = Sb3VecEnvWrapper(env, fast_variant=args_cli.fast, keep_info=not args_cli.no_info)

    # if "ppo" not in args_cli.algo:
    #     env = RescaleActionWrapper(env, percent=5.0)
    from isaaclab_rl.sb3 import ClipActionWrapper

    # For Unitree A1/GO1/... (action_scale=0.25)
    # env = ClipActionWrapper(env, percent=5)
    env = ClipActionWrapper(env, percent=3)
    # For Anymal
    # env = ClipActionWrapper(env, percent=2.5)
    # from isaaclab_rl.sb3 import ClipActionWrapper

    # env = ClipActionWrapper(env, percent=3.0)
    # from isaaclab_rl.sb3 import PenalizeCloseToBoundWrapper
    # env = PenalizeCloseToBoundWrapper(env, min_dist=0.5, max_cost=1.0)

    # From PPO Run
    # from isaaclab_rl.sb3 import ClipActionWrapper

    # low = np.array([-3.6, -2.5, -3.1, -1.8, -4.5, -4.2, -4.0, -3.9, -2.8, -2.8, -2.9, -2.7])
    # high = np.array([3.2, 2.8, 2.7, 2.8, 2.9, 2.7, 3.2, 2.9, 7.2, 5.7, 5.0, 5.8])
    # env = ClipActionWrapper(env, low=low.astype(np.float32), high=high.astype(np.float32))

    print(f"Action space: {env.action_space}")

    if args_cli.storage and args_cli.study_name:
        print("Loading from Optuna study...")
        hyperparams = load_trial(args_cli.storage, args_cli.study_name, args_cli.trial_id)
        agent_cfg.update(hyperparams)

    if args_cli.hyperparams is not None:
        print("Updating hyperparams from cli")
        agent_cfg.update(args_cli.hyperparams)

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

    # agent_cfg["param_resets"] = [int(i * 4e7) for i in range(1, 10)]
    # agent_cfg["policy_kwargs"]["squash_output"] = squash_output
    # agent_cfg["policy_kwargs"]["optimizer_class"] = optax.adam
    # agent_cfg["policy_kwargs"]["optimizer_kwargs"] = {"eps": 1e-5}
    # agent_cfg["policy_kwargs"]["ortho_init"] = True
    # agent_cfg["policy_kwargs"]["net_arch"] = [512, 256, 128]

    # Sort for printing
    hyperparams = {key: agent_cfg[key] for key in sorted(agent_cfg.keys())}

    pprint(hyperparams)

    saved_hyperparams = deepcopy(hyperparams)
    try:
        dump_yaml(os.path.join(log_dir, "hyperparams.yaml"), saved_hyperparams)
    except ValueError:
        # Backward compat with elu not being JIT compatible
        del saved_hyperparams["policy_kwargs"]["activation_fn"]
        dump_yaml(os.path.join(log_dir, "hyperparams.yaml"), saved_hyperparams)

    agent_cfg["tensorboard_log"] = log_dir

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    algo_class = {
        "ppo_sb3": sb3.PPO,
        "ppo": sbx.PPO,
        "tqc": sbx.TQC,
        "sac": sbx.SAC,
    }[args_cli.algo]

    agent = algo_class(env=env, verbose=1, **agent_cfg)

    print(f"{env.num_envs=}")
    # callbacks for agent
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=log_dir,
        name_prefix="model",
        verbose=2,
        save_vecnormalize=True,
    )
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]

    # train the agent
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=None,
        )

    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
