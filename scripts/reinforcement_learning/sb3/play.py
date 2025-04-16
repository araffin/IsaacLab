# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py
# --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 50 --algo tqc --fast --use_last_checkpoint


"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import logging
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the algorithm.", choices=["ppo", "sac", "tqc"])
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--fast", action="store_true", default=False, help="Faster correct training but not extras logged.")
parser.add_argument("--plot-action-dist", action="store_true", default=False, help="Plot action distribution.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

import sbx

# from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # directory for logging into
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.algo, args_cli.task))
    log_root_path = os.path.abspath(log_root_path)
    # checkpoint and log_dir stuff
    if args_cli.use_pretrained_checkpoint:
        checkpoint_path = get_published_pretrained_checkpoint("sb3", args_cli.task)
        if not checkpoint_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # FIXME: last checkpoint doesn't seem to really use the last one'
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    agent_cfg = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=args_cli.fast)

    # Plot action taken
    if args_cli.plot_action_dist:
        from isaaclab_rl.sb3 import PlotActionVecEnvWrapper

        env = PlotActionVecEnvWrapper(env, plot_freq=1_000)

    maybe_action_space = Path(log_dir) / "action_space.txt"
    if maybe_action_space.is_file():
        space_str = maybe_action_space.read_text()
        if "[" in space_str:
            low_str, high_str, *_ = space_str.split(", ")
            low = np.array(eval(low_str.replace("Box(", "").replace("  ", " ").replace(" ", ",")))
            high = np.array(eval(high_str.replace("  ", " ").replace(" ", ",")))
        else:
            low, high = None, None
    else:
        low = np.array([-2.0, -0.4, -2.6, -1.3, -2.2, -1.9, -0.7, -0.4, -2.1, -2.4, -2.5, -1.7])
        high = np.array([1.1, 2.6, 0.7, 1.9, 1.3, 2.6, 3.4, 3.8, 3.4, 3.4, 1.9, 2.1])
        # low = np.array([-2.3, -0.8, -2.9, -1.7, -2.7, -2.8, -1.2, -0.9, -2.9, -3.2, -3.2, -2.1])
        # high = np.array([1.4, 2.9, 1.1, 2.3, 1.8, 3.1, 3.9, 4.1, 4.3, 4. , 2.7, 3. ])

    if low is not None and high is not None:
        from isaaclab_rl.sb3 import ClipActionWrapper

        # env = ClipActionWrapper(env, percent=3)
        env = ClipActionWrapper(env, low=low.astype(np.float32), high=high.astype(np.float32))

    print(f"Action space: {env.action_space}")

    vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
    vec_norm_path = Path(vec_norm_path)

    logging.getLogger().setLevel(logging.INFO)
    # normalize environment (if needed)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
    elif "normalize_input" in agent_cfg:
        print("Relearning normalization")
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")

    # import stable_baselines3 as sb3
    algo_class = {"ppo": sbx.PPO, "sac": sbx.SAC, "tqc": sbx.TQC}[args_cli.algo]

    agent = algo_class.load(checkpoint_path, env, print_system_info=True)

    # agent.policy._squash_output = False

    dt = env.unwrapped.step_dt

    # from stable_baselines3.common.evaluation import evaluate_policy

    # for idx in range(1, 21):
    #     import time
    #     start_time = time.time()
    #     # env.seed(3)
    #     mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=50, warn=False)
    #     dt = time.time() - start_time
    #     print(f"Eval {idx}: {mean_reward:.2f} +/ {std_reward:.2f}, took {dt:.1f}s")

    # reset environment
    obs = env.reset()
    timestep = 0
    current_rewards = np.zeros(args_cli.num_envs)
    current_lengths = np.zeros(args_cli.num_envs)
    log_returns = []
    log_length = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)

        # print(np.max(actions), np.min(actions))

        current_rewards += rewards
        current_lengths += 1
        log_returns = np.concatenate((log_returns, current_rewards[dones]))
        log_length = np.concatenate((log_length, current_lengths[dones]))
        current_rewards[dones] = 0.0
        current_lengths[dones] = 0.0
        # Report performance
        if len(log_returns) > 200:
            print(f"Mean reward: {np.mean(log_returns):.2f} +/- {np.std(log_returns):.2f}")
            print(f"Mean length: {np.mean(log_length):.2f} +/- {np.std(log_length):.2f}")
            log_returns = []
            log_length = []

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
