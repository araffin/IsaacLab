# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an environment instance to Stable-Baselines3 vectorized environment.

The following example shows how to wrap an environment for Stable-Baselines3:

.. code-block:: python

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)

"""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
import warnings
from gymnasium import spaces
from typing import Any

import jax.numpy as jnp
import seaborn as sns
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class LogCallback(BaseCallback):
    """
    Log additional data (like curriculum level)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if len(self.locals["infos"]) > 0:
            if "Curriculum/terrain_levels" in self.locals["infos"][0]:
                curriculum_level = self.locals["infos"][0]["Curriculum/terrain_levels"]
                self.logger.record("curriculum/terrain_levels", curriculum_level)
            for key, value in self.locals["infos"][0].items():
                if "Episode_Reward/" in key or "Metrics/" in key or "Episode_Termination" in key:
                    self.logger.record(key.lower(), value)
        return True


# TO BE able to pickle it, no JIT
# https://github.com/jax-ml/jax/blob/72e7b93b4d0648ade1fc96bd0dad946493a4fe2d/jax/_src/nn/functions.py#L294
def elu(x, alpha=1.0):
    safe_x = jnp.where(x > 0, 0.0, x)
    return jnp.where(x > 0, x, alpha * jnp.expm1(safe_x))


# For plotting
sns.set_theme()


class PlotActionVecEnvWrapper(VecEnvWrapper):
    """
    VecEnv wrapper for plotting the taken actions.
    """

    def __init__(self, venv, plot_freq: int = 10_000):
        super().__init__(venv)
        # Action buffer
        assert isinstance(self.action_space, spaces.Box)
        self.n_actions = self.action_space.shape[0]
        self.actions = np.zeros((plot_freq, self.num_envs, self.n_actions))
        self.n_steps = 0
        self.plot_freq = plot_freq

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, dones, infos

    def step_async(self, actions):
        self.actions[self.n_steps % self.plot_freq] = actions
        self.n_steps += 1
        if self.n_steps % self.plot_freq == 0:
            self.plot()
        self.venv.step_async(actions)

    def plot(self) -> None:
        # Flatten the env dimension
        actions = self.actions.reshape(-1, self.n_actions)
        np.set_printoptions(precision=1)
        print(f"{np.percentile(actions, 2.5, axis=0)!r}")
        print(f"{np.percentile(actions, 97.5, axis=0)!r}")
        print("===")
        print(f"{np.percentile(actions, 0.5, axis=0)!r}")
        print(f"{np.percentile(actions, 99.5, axis=0)!r}")
        print("Saving to /tmp/plot_actions.npz")
        np.savez("/tmp/plot_actions.npz", actions=actions)
        n_steps = self.num_envs * self.n_steps
        # Create a figure with subplots for each action dimension
        n_rows = min(2, self.n_actions // 2 + 1)
        n_cols = max(self.n_actions // 2, 1)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
        fig.suptitle(f"Distribution of Actions per Dimension after {n_steps} steps", fontsize=16)

        # Flatten the axes array for easy iteration
        if n_rows > 1:
            axes = axes.flatten()
        else:
            # Special case, n_actions == 1
            axes = [axes]

        # Plot the distribution for each action dimension
        for i in range(self.n_actions):
            sns.histplot(actions[:, i], kde=True, ax=axes[i], stat="density")
            axes[i].set_title(f"Action Dimension {i+1}")
            axes[i].set_xlabel("Action Value")
            axes[i].set_ylabel("Density")

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()


# remove SB3 warnings because PPO with bigger net actually benefits from GPU
warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")

"""
Configuration Parser.
"""


def process_sb3_cfg(cfg: dict, num_envs: int) -> dict:
    """Convert simple YAML types to Stable-Baselines classes/components.

    Args:
        cfg: A configuration dictionary.
        num_envs: the number of parallel environments (used to compute `batch_size` for a desired number of minibatches)

    Returns:
        A dictionary containing the converted configuration.

    Reference:
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/utils/exp_manager.py#L358
    """

    def update_dict(hyperparams: dict[str, Any]) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in ["policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"]:
                    hyperparams[key] = eval(value)
                elif key in ["learning_rate", "clip_range", "clip_range_vf"]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[key] = lambda progress_remaining: progress_remaining * initial_value
                    elif isinstance(value, (float, int)):
                        # negative value: ignore (ex: for clipping)
                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    else:
                        raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")

        # Convert to a desired batch_size (n_steps=2048 by default for SB3 PPO)
        if "n_minibatches" in hyperparams:
            hyperparams["batch_size"] = (hyperparams.get("n_steps", 2048) * num_envs) // hyperparams["n_minibatches"]
            del hyperparams["n_minibatches"]

        return hyperparams

    # parse agent configuration and convert to classes
    return update_dict(cfg)


"""
Vectorized environment wrapper.
"""


class Sb3VecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for Stable Baselines3.

    Isaac Sim internally implements a vectorized environment. However, since it is
    still considered a single environment instance, Stable Baselines tries to wrap
    around it using the :class:`DummyVecEnv`. This is only done if the environment
    is not inheriting from their :class:`VecEnv`. Thus, this class thinly wraps
    over the environment from :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.

    Note:
        While Stable-Baselines3 supports Gym 0.26+ API, their vectorized environment
        uses their own API (i.e. it is closer to Gym 0.21). Thus, we implement
        the API for the vectorized environment.

    We also add monitoring functionality that computes the un-discounted episode
    return and length. This information is added to the info dicts under key `episode`.

    In contrast to the Isaac Lab environment, stable-baselines expect the following:

    1. numpy datatype for MDP signals
    2. a list of info dicts for each sub-environment (instead of a dict)
    3. when environment has terminated, the observations from the environment should correspond
       to the one after reset. The "real" final observation is passed using the info dicts
       under the key ``terminal_observation``.

    .. warning::

        By the nature of physics stepping in Isaac Sim, it is not possible to forward the
        simulation buffers without performing a physics step. Thus, reset is performed
        inside the :meth:`step()` function after the actual physics step is taken.
        Thus, the returned observations for terminated environments is the one after the reset.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:

    1. https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    2. https://stable-baselines3.readthedocs.io/en/master/common/monitor.html

    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, fast_variant: bool = True):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.
            fast_variant: Use fast variant for processing info
                (Only episodic reward, lengths and truncation info are included)
        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.fast_variant = fast_variant
        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode
        self.observation_processors = {}
        self._process_spaces()
        # add buffer for logging episodic information
        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_episode_rewards(self) -> list[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.tolist()

    def get_episode_lengths(self) -> list[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.tolist()

    """
    Operations - MDP
    """

    def seed(self, seed: int | None = None) -> list[int | None]:  # noqa: D102
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

    def reset(self) -> VecEnvObs:  # noqa: D102
        obs_dict, _ = self.env.reset()
        # reset episodic information buffers
        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)
        # convert data types to numpy depending on backend
        return self._process_obs(obs_dict)

    def step_async(self, actions):  # noqa: D102
        # convert input to numpy array
        if not isinstance(actions, torch.Tensor):
            actions = np.asarray(actions)
            actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        # convert to tensor
        self._async_actions = actions

    def step_wait(self) -> VecEnvStepReturn:  # noqa: D102
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(self._async_actions)
        # compute reset ids
        dones = terminated | truncated

        # convert data types to numpy depending on backend
        # note: ManagerBasedRLEnv uses torch backend (by default).
        obs = self._process_obs(obs_dict)
        rewards = rew.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()

        self.reset_ids = reset_ids = dones.nonzero()[0]

        # update episode un-discounted return and length
        self._ep_rew_buf += rewards
        self._ep_len_buf += 1
        # convert extra information to list of dicts
        infos = self._process_extras(obs, terminated, truncated, extras, reset_ids)

        # reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0.0
        self._ep_len_buf[reset_ids] = 0

        return obs, rewards, dones, infos

    def close(self):  # noqa: D102
        self.env.close()

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        # obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        if method_name == "render":
            # gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # this isn't properly implemented but it is not necessary.
            # mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        # fake implementation to be able to use `evaluate_policy()` helper
        return [False]

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")

    """
    Helper functions.
    """

    def _process_spaces(self):
        # process observation space
        observation_space = self.unwrapped.single_observation_space["policy"]
        if isinstance(observation_space, gym.spaces.Dict):
            for obs_key, obs_space in observation_space.spaces.items():
                processors: list[callable[[torch.Tensor], Any]] = []
                # assume normalized, if not, it won't pass is_image_space, which check [0-255].
                # for scale like image space that has right shape but not scaled, we will scale it later
                if is_image_space(obs_space, check_channels=True, normalized_image=True):
                    actually_normalized = np.all(obs_space.low == -1.0) and np.all(obs_space.high == 1.0)
                    if not actually_normalized:
                        if np.any(obs_space.low != 0) or np.any(obs_space.high != 255):
                            raise ValueError(
                                "Your image observation is not normalized in environment, and will not be"
                                "normalized by sb3 if its min is not 0 and max is not 255."
                            )
                        # sb3 will handle normalization and transpose, but sb3 expects uint8 images
                        if obs_space.dtype != np.uint8:
                            processors.append(lambda obs: obs.to(torch.uint8))
                        observation_space.spaces[obs_key] = gym.spaces.Box(0, 255, obs_space.shape, np.uint8)
                    else:
                        # sb3 will NOT handle the normalization, while sb3 will transpose, its transpose applies to all
                        # image terms and maybe non-ideal, more, if we can do it in torch on gpu, it will be faster then
                        # sb3 transpose it in numpy with cpu.
                        if not is_image_space_channels_first(obs_space):

                            def tranp(img: torch.Tensor) -> torch.Tensor:
                                return img.permute(2, 0, 1) if len(img.shape) == 3 else img.permute(0, 3, 1, 2)

                            processors.append(tranp)
                            h, w, c = obs_space.shape
                            observation_space.spaces[obs_key] = gym.spaces.Box(-1.0, 1.0, (c, h, w), obs_space.dtype)

                    def chained_processor(obs: torch.Tensor, procs=processors) -> Any:
                        for proc in procs:
                            obs = proc(obs)
                        return obs

                    # add processor to the dictionary
                    if len(processors) > 0:
                        self.observation_processors[obs_key] = chained_processor

        # obtain gym spaces
        # note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.
        action_space = self.unwrapped.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            print(f"Overriding action space {action_space} to Box(low=-100, high=100) because it is unbounded")
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)

        # initialize vec-env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"
        obs = obs_dict["policy"]
        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            for key, value in obs.items():
                if key in self.observation_processors:
                    obs[key] = self.observation_processors[key](value)
                obs[key] = obs[key].detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs

    def _process_extras(
        self, obs: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, extras: dict, reset_ids: np.ndarray
    ) -> list[dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""
        # faster version: only process env that terminated and add bootstrapping info
        if self.fast_variant:
            infos = [{} for _ in range(self.num_envs)]

            for idx in reset_ids:
                # fill-in episode monitoring info
                infos[idx]["episode"] = {
                    "r": self._ep_rew_buf[idx],
                    "l": self._ep_len_buf[idx],
                }

                # fill-in bootstrap information
                infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]

                # add information about terminal observation separately
                if isinstance(obs, dict):
                    terminal_obs = {key: value[idx] for key, value in obs.items()}
                else:
                    terminal_obs = obs[idx]
                infos[idx]["terminal_observation"] = terminal_obs

            # Log curriculum (already a mean, stored in the env with index=0)
            if "log" in extras.keys():
                if "Curriculum/terrain_levels" in extras["log"]:
                    infos[0]["Curriculum/terrain_levels"] = extras["log"]["Curriculum/terrain_levels"]
                # Log additional metrics
                for key, value in extras["log"].items():
                    if "Episode_Reward/" in key or "Metrics/" in key or "Episode_Termination" in key:
                        try:
                            infos[0][key] = value.item()
                        except AttributeError:
                            # Already a float
                            infos[0][key] = value

            return infos

        # create empty list of dictionaries to fill
        infos: list[dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.num_envs)]

        # fill-in information for each sub-environment
        # note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = dict()
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None

            # fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # fill-in information from extras
            for key, value in extras.items():
                # 1. remap extra episodes information safely
                # 2. for others just store their values
                if key == "log":
                    # only log this data for episodes that are terminated
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]
            # add information about terminal observation separately
            if idx in reset_ids:
                # extract terminal observations
                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                # add info to dict
                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None
        # return list of dictionaries
        return infos


class RescaleActionWrapper(VecEnvWrapper):

    def __init__(self, vec_env, percent=5.0, scheduler=None):
        super().__init__(vec_env)
        self.low, self.high = vec_env.action_space.low, vec_env.action_space.high
        self.percent = percent
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=vec_env.action_space.shape,
            dtype=np.float32,
        )
        self.n_steps = 0
        self.scheduler = scheduler
        if scheduler:
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(vec_env.observation_space.shape[0] + 1,),
                dtype=np.float32,
            )

    def add_to_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.scheduler:
            return obs
        if len(obs.shape) > 1:
            return np.concatenate((obs, np.full((self.num_envs, 1), self.percent / 100.0)), axis=1)
        return np.concatenate((obs, np.full((1,), self.percent / 100.0)))

    def step_async(self, actions: np.ndarray) -> None:
        self.n_steps += self.num_envs
        if self.scheduler:
            self.percent = self.scheduler(self.n_steps)

        # Rescale the action from [-1, 1] to [low, high]
        low = self.percent * 0.01 * self.low
        high = self.percent * 0.01 * self.high
        rescaled_action = low + (0.5 * (actions + 1.0) * (high - low))
        self.venv.step_async(rescaled_action)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return self.add_to_obs(obs)

    def step_wait(self):
        obs, reward, done, infos = self.venv.step_wait()
        for idx in self.venv.reset_ids:
            infos[idx]["terminal_observation"] = self.add_to_obs(infos[idx]["terminal_observation"])

        return self.add_to_obs(obs), reward, done, infos


class ClipActionWrapper(VecEnvWrapper):

    def __init__(self, vec_env, percent=5.0, low=None, high=None):
        super().__init__(vec_env)
        self.low, self.high = vec_env.action_space.low, vec_env.action_space.high
        self.percent = percent
        print(f"Action low before: {self.low}")

        self.action_space = gym.spaces.Box(
            low=low if low is not None else self.percent * 0.01 * self.low,
            high=high if high is not None else self.percent * 0.01 * self.high,
            shape=vec_env.action_space.shape,
            dtype=np.float32,
        )
        print(f"Action low after: {self.action_space.low}")
        print(f"Action high after: {self.action_space.high}")

    def step_async(self, actions: np.ndarray) -> None:
        # Clipping is done inside the algorithm
        # self.venv.step_async(np.clip(actions, self.action_space.low, self.action_space.high))
        self.venv.step_async(actions)

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


class PenalizeCloseToBoundWrapper(VecEnvWrapper):

    def __init__(self, vec_env, min_dist: float = 1.0, max_cost: float = 1.0):
        super().__init__(vec_env)
        self.n_actions = self.action_space.shape[0]
        self.low = self.action_space.low + min_dist
        self.high = self.action_space.high - min_dist
        self.coeff = max_cost
        self.min_dist = min_dist

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.venv.step_async(actions)

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        delta_low = (self.actions - self.low).min(axis=1)
        delta_high = (self.high - self.actions).min(axis=1)
        too_close_low = (delta_low < 0.0).nonzero()[0]
        too_close_high = (delta_high < 0.0).nonzero()[0]
        reward[too_close_low] -= self.coeff * (delta_low[too_close_low] / self.min_dist) ** 2
        reward[too_close_high] -= self.coeff * (delta_low[too_close_high] / self.min_dist) ** 2

        return obs, reward, done, info


def load_trial(storage: str, study_name: str, trial_id: int | None = None, convert: bool = True) -> dict[str, Any]:
    import optuna

    optuna_storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(storage))
    study = optuna.load_study(storage=optuna_storage, study_name=study_name)
    if trial_id is not None:
        params = study.trials[trial_id].params
    else:
        params = study.best_trial.params

    if convert:
        return to_hyperparams(params)
    return params


def to_hyperparams(sampled_params: dict[str, Any]) -> dict[str, Any]:
    import optax

    hyperparams = sampled_params.copy()

    if "ent_coef_init" in hyperparams:
        hyperparams["ent_coef"] = f"auto_{hyperparams['ent_coef_init']}"
        del hyperparams["ent_coef_init"]

    if "one_minus_gamma" in hyperparams:
        hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
        del hyperparams["one_minus_gamma"]

    if "net_arch_complexity" in sampled_params:
        idx = sampled_params["net_arch_complexity"]
        net_arch = [
            "default",
            "medium",
            "simba",
            "large",
            "xlarge",
        ][idx]
        del hyperparams["net_arch_complexity"]
    else:
        net_arch = sampled_params["net_arch"]
        del hyperparams["net_arch"]

    for name in ["batch_size", "gradient_steps", "policy_delay", "train_freq"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    policy = "SimbaPolicy" if net_arch == "simba" else "MlpPolicy"

    net_arch = {
        "default": [256, 256],
        "medium": [128, 128, 128],
        "simba": {
            "pi": [128, 128],
            "qf": [256, 256],
        },
        "large": [512, 256, 128],
        "xlarge": [512, 512, 256],
    }[net_arch]
    # activation_fn = {
    #     "elu": flax.linen.elu,
    #     "relu": flax.linen.relu,
    #     "gelu": flax.linen.gelu,
    # }[sampled_params["activation_fn"]]
    if "activation_fn" in hyperparams:
        del hyperparams["activation_fn"]
    if "learning_starts" in hyperparams:
        del hyperparams["learning_starts"]

    log_std_init = 0.0
    squash_output = True
    ortho_init = False
    if "log_std_init" in sampled_params:
        log_std_init = sampled_params["log_std_init"]
        squash_output = False
        ortho_init = True
        del hyperparams["log_std_init"]

    return {
        "policy": policy,
        "buffer_size": 800_000,
        "learning_starts": 2_000,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": elu,
            "optimizer_class": optax.adamw,
            "layer_norm": True,
            "log_std_init": log_std_init,
            "squash_output": squash_output,
            "ortho_init": ortho_init,
        },
        **hyperparams,
    }
