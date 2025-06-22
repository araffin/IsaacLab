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

from isaaclab_rl.sb3 import LogCallback, LogEveryNTimesteps, Sb3VecEnvWrapper, elu, load_trial, process_sb3_cfg

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import disney_bdx.tasks  # noqa: F401

from isaaclab_tasks.utils.hydra import hydra_task_config

ppo_defaults = dict(
    policy="MlpPolicy",
    # n_steps=25,
    # batch_size=6400,  # for 1024 envs, to have 4 minibatches
    n_steps=24,
    # 25600 for 25 steps
    batch_size=24576,  # 4 mini-batches for 4096 envs
    # target_kl=0.01,
    gae_lambda=0.95,
    n_epochs=5,
    # ent_coef=0.01,
    ent_coef=0.005,  # For Anymal-C env
    learning_rate=1e-3,
    clip_range=0.2,
    vf_coef=1.0,
    max_grad_norm=1.0,
    policy_kwargs=dict(
        activation_fn=elu,
        net_arch=[512, 256, 128],
        # Match PyTorch Implementation
        optimizer_kwargs=dict(eps=0.0, eps_root=1e-8),
        # optimizer_kwargs=dict(eps=1e-6),
        # optimizer_class=optax.adamw,
        # log_std_init=-0.8,
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
        net_arch=[512, 256, 128],
        optimizer_kwargs=dict(eps=1e-8),
        # optimizer_kwargs=dict(eps=1e-5),
        ortho_init=False,
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

# Optimized with TQC on A1 flat for 1024 envs
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
    # qf_learning_rate=7e-4,
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
            # "n_timesteps": 5e7,
            "n_timesteps": 15e7,
            # Note: no normalization for Anymal Rough env
            "normalize_input": "Rough" not in args_cli.task,
            # "normalize_input": True,
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
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
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

    low = None
    if "Unitree-A" in args_cli.task or "Unitree-Go" in args_cli.task:
        # For Unitree A1/GO1/... (action_scale=0.25)
        multiplier = 1.25 if "Rough-" in args_cli.task else 1.0
        # From rough:
        # low = multiplier * np.array([-1.8, -2.3, -1.2, -0.8, -2.5, -3.1, -1.2, -1.4, -1.8, -1.1, -1.2, -1.5])
        # high = multiplier * np.array([1.5, 2.0, 1.8, 2.6, 1.5, 1.4, 2.5, 2.5, 3.0, 3.2, 2.7, 3.2])
        low = multiplier * np.array([-2.0, -0.4, -2.6, -1.3, -2.2, -1.9, -0.7, -0.4, -2.1, -2.4, -2.5, -1.7])
        high = multiplier * np.array([1.1, 2.6, 0.7, 1.9, 1.3, 2.6, 3.4, 3.8, 3.4, 3.4, 1.9, 2.1])
    elif "-Anymal" in args_cli.task:
        # Anymal-C Rough
        # low = 1.2 * np.array([-1.4, -1.2, -0.5, -0.7, -1.7, -1.4, -1.3, -1.3, -2.3, -1.7, -1.8, -2.0])
        # high = 1.2 * np.array([1.0, 1.0, 1.5, 1.2, 1.1, 1.4, 1.6, 1.1, 2.2, 1.6, 1.3, 2.1])
        low = 1.25 * np.array([-1.3, -1.4, -0.2, -0.2, -0.7, -1.7, -0.6, -1.6, -2.8, -1.1, -2.9, -1.0])
        high = 1.25 * np.array([0.2, 0.6, 1.5, 1.3, 1.6, 0.6, 1.9, 0.7, 0.7, 2.4, 0.7, 2.3])
    elif "-Disney-Bdx" in args_cli.task:
        # low = np.full(len(env.action_space.low), -3.0)
        # high = np.full(len(env.action_space.low), 3.0)
        low = np.array([-0.6, -0.5, -0.4, -0.5, -0.7, -0.2, -0.1, -0.6, -1.1, -1.5, -0.5, -0.6, -0.8, -1.0, -0.7, -0.2])
        high = np.array([0.4, 0.3, 0.5, -0.1, 0.6, 0.3, 1.3, 1.1, 0.0, 0.7, 0.7, 1.5, 0.4, 0.6, 0.7, 0.8])

    if "ppo" not in args_cli.algo and low is not None:
        env = ClipActionWrapper(env, low=low.astype(np.float32), high=high.astype(np.float32))

    (Path(log_dir) / "action_space.txt").write_text(str(env.action_space))

    # from isaaclab_rl.sb3 import PenalizeCloseToBoundWrapper

    # min_dist, max_cost = 0.5, 0.5
    # print(f"{min_dist=}, {max_cost=}")

    # env = PenalizeCloseToBoundWrapper(env, min_dist=min_dist, max_cost=max_cost)

    print(f"Action space: {env.action_space}")

    if args_cli.storage and args_cli.study_name:
        print("Loading from Optuna study...")
        hyperparams = load_trial(args_cli.storage, args_cli.study_name, args_cli.trial_id)
        agent_cfg.update(hyperparams)

    # Special: squash output and log_std_init
    # agent_cfg["policy_kwargs"]["squash_output"] = False
    # agent_cfg["policy_kwargs"]["ortho_init"] = True
    # agent_cfg["policy_kwargs"]["log_std_init"] = -0.5
    # agent_cfg["policy_kwargs"]["net_arch"] = [1024, 512, 256]
    # agent_cfg["param_resets"] = [int(i * 4e7) for i in range(1, 10)]
    # agent_cfg["policy_kwargs"]["squash_output"] = squash_output
    # agent_cfg["policy_kwargs"]["optimizer_class"] = optax.adam
    # weight_decay:0.001 (0.0001 by default)
    # agent_cfg["policy_kwargs"]["optimizer_kwargs"] = {"eps": 1e-5, "b1": 0.5} # b1=0.5 (default 0.9)
    # agent_cfg["policy_kwargs"]["ortho_init"] = True
    if args_cli.hyperparams is not None:
        print("Updating hyperparams from cli")
        agent_cfg.update(args_cli.hyperparams)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    vec_norm_path = None
    # vec_norm_path = Path("./logs/model_vecnormalize.pkl")

    if vec_norm_path and vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        # Constant norm
        env.training = False
        env.norm_reward = False

    elif norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

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

    if "ppo" not in args_cli.algo and agent_cfg.get("lr_schedule") is not None:
        from stable_baselines3.common.utils import LinearSchedule

        agent_cfg["learning_rate"] = LinearSchedule(start=5e-4, end=1e-5, end_fraction=0.15)
        print(agent_cfg["learning_rate"])
        del agent_cfg["lr_schedule"]

    # from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    # n_actions = env.action_space.shape[0]
    # noise_std = 0.2
    # agent_cfg["action_noise"] = NormalActionNoise(
    #     mean=np.zeros(n_actions),
    #     sigma=np.full(n_actions, noise_std),
    # )
    # from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
    # n_actions = env.action_space.shape[0]
    # noise_std = 0.3
    # agent_cfg["action_noise"] = OrnsteinUhlenbeckActionNoise(
    #     mean=np.zeros(n_actions),
    #     sigma=np.full(n_actions, noise_std),
    # )
    # print(agent_cfg["action_noise"])

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
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval), LogCallback()]

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
