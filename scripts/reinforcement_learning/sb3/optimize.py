# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ./isaaclab.sh -p scripts/reinforcement_learning/sb3/optimize.py --task Isaac-Velocity-Flat-Unitree-A1-v0
#  --num_envs 2048 --headless --algo tqc --seed 3 --storage logs/sb3_tqc_flat_a1.log


"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import gc
import signal
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the algorithm.", choices=["ppo", "sac", "tqc"])
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--no-info",
    action="store_true",
    default=False,
    help="Fastest and incorrect training but no statistics.",
)
parser.add_argument(
    "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
)
parser.add_argument("--n-trials", help="Max number of trials for this process", type=int, default=1000)
parser.add_argument("-name", "--study-name", help="Study name for distributed optimization", type=str, default=None)
parser.add_argument(
    "--sampler",
    help="Sampler to use when optimizing hyperparameters",
    type=str,
    default="auto",
    choices=["random", "tpe", "cmaes", "auto"],
)
parser.add_argument("--pop-size", help="Initial population size for CMAES", type=int, default=10)
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


def raise_interrupt(*args):
    raise KeyboardInterrupt


# Disable KeyboardInterrupt override
signal.signal(signal.SIGINT, raise_interrupt)


"""Rest everything follows."""

import gymnasium as gym
import random
import time

import optuna
import optunahub
import sbx
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

# from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, to_hyperparams

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


class TimeoutCallback(BaseCallback):
    def __init__(self, timeout: int = 60, start_after: int = 0, verbose: int = 0):
        super().__init__(verbose)
        # Timeout in second
        self.timeout = timeout
        self.start_time = None
        # Wait for JIT
        self.start_after = start_after

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start_after:
            return True

        if not self.start_time:
            self.start_time = time.time()

        if (time.time() - self.start_time) > self.timeout:
            return False
        return True


def sample_tqc_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sampler for TQC hyperparameters."""
    # From 0.975 to 0.995
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.005, 0.025, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.002, log=True)
    # qf_learning_rate = trial.suggest_float("qf_learning_rate", 1e-5, 0.01, log=True)
    ent_coef_init = trial.suggest_float("ent_coef_init", 0.001, 0.02, log=True)
    # From 128 to 2*12 = 4096
    batch_size_pow = trial.suggest_int("batch_size_pow", 7, 12, log=True)
    # net_arch = trial.suggest_categorical("net_arch", ["default", "medium", "simba", "large", "xlarge"])
    # Use int to be able to use CMA-ES
    net_arch_complexity = trial.suggest_int("net_arch_complexity", 3, 4)
    # activation_fn = trial.suggest_categorical("activation_fn", ["elu", "relu", "gelu"])
    # From 1 to 8
    train_freq_pow = trial.suggest_int("train_freq_pow", 0, 3)
    # From 1 to 1024
    gradient_steps_pow = trial.suggest_int("gradient_steps_pow", 0, 10)
    # gradient_steps = trial.suggest_categorical("gradient_steps", [64, 128, 256, 512])
    # learning_starts = trial.suggest_categorical("learning_starts", [100, 1000, 2000])
    # From 1 to 32
    policy_delay_pow = trial.suggest_int("policy_delay_pow", 0, 5)
    # Polyak coeff
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    # For Gaussian actor
    log_std_init = trial.suggest_float("log_std_init", -2.5, 0.0)

    # Display true values
    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2**batch_size_pow)
    trial.set_user_attr("gradient_steps", 2**gradient_steps_pow)
    trial.set_user_attr("policy_delay", 2**policy_delay_pow)
    trial.set_user_attr("train_freq", 2**train_freq_pow)

    return to_hyperparams({
        "train_freq_pow": train_freq_pow,
        "gradient_steps_pow": gradient_steps_pow,
        "batch_size_pow": batch_size_pow,
        "tau": tau,
        # "learning_starts": learning_starts,
        "one_minus_gamma": one_minus_gamma,
        "learning_rate": learning_rate,
        # "qf_learning_rate": qf_learning_rate,
        "policy_delay_pow": policy_delay_pow,
        "ent_coef_init": ent_coef_init,
        "net_arch_complexity": net_arch_complexity,
        # "activation_fn": activation_fn,
        # "optimizer_class": optax.adamw,
        "log_std_init": log_std_init,
    })


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
    # run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_root_path = os.path.abspath(os.path.join("logs", "sb3_optim", args_cli.algo, args_cli.task))
    # print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # print(f"Exact experiment name requested from command line: {run_info}")
    # log_dir = os.path.join(log_root_path, run_info)

    # post-process agent configuration
    # agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    # policy_arch = agent_cfg.pop("policy")
    # n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=True, keep_info=not args_cli.no_info)

    # if "ppo" not in args_cli.algo:
    #     from isaaclab_rl.sb3 import ClipActionWrapper
    #     env = ClipActionWrapper(env, percent=3)

    print(f"Action space: {env.action_space}")

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
    print(f"{env.num_envs=}")

    N_STARTUP_TRIALS = 5

    storage = args_cli.storage
    if storage is not None and storage.endswith(".log"):
        # Create folder if it doesn't exist
        Path(storage).parent.mkdir(parents=True, exist_ok=True)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(args_cli.storage),
        )

    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = {
        "auto": optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=args_cli.seed),
        "tpe": TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True, seed=args_cli.seed),
        "cmaes": CmaEsSampler(
            seed=args_cli.seed,
            restart_strategy="bipop",
            popsize=args_cli.pop_size,
            n_startup_trials=N_STARTUP_TRIALS,
        ),
        "random": RandomSampler(seed=args_cli.seed),
    }[args_cli.sampler]

    study = optuna.create_study(
        study_name=args_cli.study_name,
        storage=storage,
        sampler=sampler,
        pruner=None,
        direction="maximize",
        load_if_exists=True,
    )

    # study.enqueue_trial(
    #     {
    #         "train_freq": 5,
    #         "gradient_steps": 512,
    #         "batch_size": 512,
    #         "learning_starts": 1_000,
    #         "one_minus_gamma": 1 - 0.985,
    #         "learning_rate": 3e-4,
    #         # "qf_learning_rate": qf_learning_rate,
    #         "policy_delay": 10,
    #         "ent_coef_init": 0.01,
    #         "net_arch": "simba",
    #         "activation_fn": "elu",
    #         # "optimizer_class": optax.adamw,
    #     },
    #     user_attrs={"memo": "best known, manually tuned"},
    # )
    # from isaaclab_rl.sb3 import load_trial

    # # Best trials
    # trial_ids = [0, 56, 120, 149, 167, 172]
    # for trial_id in trial_ids:
    #     hyperparams = load_trial("logs/sb3_tqc_flat_a1.log", "tqc_flat_a1_2", trial_id=trial_id, convert=False)
    #     # Convert search space
    #     for key in ["learning_starts"]:
    #         del hyperparams[key]
    #     hyperparams["net_arch_complexity"] = {
    #         "default": 0,
    #         "medium": 1,
    #         "simba": 2,
    #     }[hyperparams["net_arch"]]
    #     del hyperparams["net_arch"]
    #     study.enqueue_trial(hyperparams, user_attrs={"memo": "from previous optim"})

    def objective(trial: optuna.Trial) -> float:
        # TODO: add support for PPO/SAC
        hyperparams = sample_tqc_params(trial)
        env.seed(args_cli.seed)
        agent = sbx.TQC(env=env, **hyperparams)
        # Start after warmup
        # optimize for best perf after 5 minutes
        callback = TimeoutCallback(timeout=60 * 5, start_after=3_000)
        agent.learn(total_timesteps=int(3e7), callback=callback)
        trial.set_user_attr("num_timesteps", agent.num_timesteps)
        env.seed(args_cli.seed)
        # do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
        mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=50, warn=False)
        trial.set_user_attr("std_reward", std_reward)

        # Free memory
        del agent.replay_buffer
        del agent.policy
        del agent
        del callback
        del hyperparams
        gc.collect()
        gc.collect()
        return mean_reward

    with contextlib.suppress(KeyboardInterrupt):
        # n_jobs=1, timeout=TIMEOUT
        study.optimize(objective, n_trials=args_cli.n_trials)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
