import copy
import math
import warnings
from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.utils import obs_as_tensor
from collections import OrderedDict
from copy import deepcopy
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import time
import sys
import torch
from torch.utils.data import DataLoader
from hybridppo.minari_helpers import MinariTransitionDataset, MultiEpisodeSequentialSampler, collate_env_batch
from functools import partial
from scipy.stats import chi2
from typing import Generator
from typing import NamedTuple, Optional, Union, Any, Callable, Dict, List, Type
from tqdm.auto import tqdm

from abc import ABC
#Instead of scaling advantage, scale returns using SimbaV2

SelfPPO = TypeVar("SelfPPO", bound="PPO")

class ExpertRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    log_prob_expert: th.Tensor  # New field for expert log probabilities

class ExpertRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super(ExpertRolloutBuffer, self).__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.log_prob_expert = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            log_prob_expert: np.ndarray,  # New arg
    ) -> None:
        super().add(obs, action, reward, episode_start, value, log_prob)
        self.log_prob_expert[self.pos-1] = log_prob_expert.clone().cpu().numpy()
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        last_gae_lam = 0
        rho_bar = 1.0  # For V-trace
        c_bar = 0.95  # For V-trace
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            ratio = np.exp(self.log_probs[step] - self.log_prob_expert[step])  # ratio = p(a|s) / p(a|s, expert)
            rho = np.clip(ratio, 1e-3, rho_bar)
            c = np.clip(ratio, 1e-3, c_bar)
            # next_ratio = np.clip(next_ratio, 1e-5, 1)
            delta = (self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step])*rho
            # last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            #For retrace
            last_gae_lam = delta + self.gamma* self.gae_lambda*next_non_terminal * last_gae_lam*c
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "log_prob_expert",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.log_prob_expert[batch_inds].flatten(),  # New
        )
        return ExpertRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class PPOExpert(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    minari_transition_dataloader = None
    minari_transition_iterator = None
    minari_transition_dataset = None

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 100,
            batch_size: int = 32,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
            rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            minari_dataset = None,
            log_prob_expert = 0
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.num_expert_envs = self.env.num_envs

        #Minari dataset supports iteration over episodes which we use in MinariTransitionDataset
        self.minari_dataset = minari_dataset
        self.minari_transition_dataset = MinariTransitionDataset(minari_dataset)
        parallel_sequential_sampler = MultiEpisodeSequentialSampler(
            self.minari_transition_dataset,
            n_envs=self.num_expert_envs,
            batch_size=n_steps,
        )
        self.minari_transition_dataloader = DataLoader(
            self.minari_transition_dataset,
            batch_sampler=parallel_sequential_sampler,
            collate_fn=partial(collate_env_batch, n_envs=self.num_expert_envs, batch_size=n_steps),
            num_workers=4,
            shuffle=False,
        )
        self.minari_transition_iterator = iter(self.minari_transition_dataloader)
        self._expert_last_obs = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]

        self.log_prob_expert = log_prob_expert


        #To make it compatible with stable baselines3 API and buffer
        #Minari Dataloader should return batch_size * n_envs in total
        #Step for online data return (1, n_envs) until n_rollout_steps
        #We need to recreate that using the dataloader
        #Each iteration over dataloader return (1, n_envs) and we run a loop until n_rollout_steps (See make_offline_rollouts)
        # self.minari_transition_dataloader = DataLoader(
        #     self.minari_transition_dataset,
        #     batch_size=self.num_expert_envs, #Will return (1, n_envs) batch
        #     shuffle=True,
        #     num_workers=4,
        #     # collate_fn=self.expert_collate_fn
        # )
        # self.minari_transition_iterator = iter(self.minari_transition_dataloader)
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.expert_buffer = ExpertRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.expert_observation_space = deepcopy(self.observation_space)
        self._expert_last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        self.expert_policy = deepcopy(self.policy)  # Copy the policy to avoid modifying the original

    def _flatten_obs(self, obs, observation_space):
        """
        Flatten a list of observations into a single dictionary.
        """
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in observation_space.spaces.keys()])

    def make_offline_rollouts(self, callback, expert_buffer: RolloutBuffer, n_rollout_steps) -> bool:
        #Read from minari dataset and fill in expert buffer
        #Minari dataset can be used with data loader
        # assert self._expert_last_obs is not None, "No previous expert observation was provided"
        if self.verbose > 0:
            print("INFO: Making offline rollouts")

        #Switch to eval mode
        self.policy.set_training_mode(False)
        n_steps = 0
        expert_buffer.reset()

        # TODO: Do we need callbacks in offline rollouts?
        # callback.update_locals(locals())
        # callback.on_rollout_start()

        # while n_steps < n_rollout_steps:
        try:
            # Batch expects 1 step from n_envs trajectories
            batch = next(self.minari_transition_iterator)
        except StopIteration:
            # Reinitialize the iterator if we run out of data
            self.minari_transition_iterator = iter(self.minari_transition_dataloader)
            batch = next(self.minari_transition_iterator) #Tensor


        for i in range(batch['observations'].shape[0]):  # Loop over timesteps
            last_obs = batch['observations'][i]
            rewards = batch['rewards'][i]
            dones = batch['dones'][i]
            actions = batch['actions'][i]
            next_obs = batch['next_observations'][i]

            # Ensure 1D shapes for per-env scalars to avoid broadcast issues
            rewards_np = rewards.squeeze(-1).cpu().numpy()
            dones_np = dones.squeeze(-1).cpu().numpy()

            # Keep episode start tracking per environment
            if not hasattr(self, "_expert_last_episode_starts"):
                self._expert_last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

            with th.no_grad():
                obs_tensor = last_obs.to(self.device)
                act_tensor = actions.to(self.device)
                _, values, log_probs = self.policy.forward_expert(obs_tensor, act_tensor)
                _, _, expert_log_probs = self.expert_policy.forward_expert(obs_tensor, act_tensor)

            expert_buffer.add(
                last_obs.cpu().numpy(),
                act_tensor.cpu().numpy(),
                rewards_np,
                self._expert_last_episode_starts,
                values.squeeze(-1),
                log_probs,
                expert_log_probs,
            )

            self._expert_last_episode_starts = dones_np

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(next_obs.to(self.device)).squeeze(-1)  # pylint: disable=unexpected-keyword-arg

        expert_buffer.compute_returns_and_advantage(
            last_values=values.cpu().numpy(),
            dones=dones_np,
        )
        if self.verbose > 0:
            print("INFO: Finished making offline rollouts")
        # callback.on_rollout_end()
        # callback.update_locals(locals())
        return True




    def train_ppo_offline(self, clip_range, clip_range_vf) -> None:
        self.policy.set_training_mode(True)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # Logging dict
        logs_online = {"ratio_current_old": [], "policy_loss": [], "value_loss": [], "entropy_loss": [],
                       "approx_kl_div": [],
                       "clip_fraction": [], "advantages": [], "log_prob": []}

        logs_offline = {"ratio_current_old": [], "policy_loss": [], "value_loss": [], "entropy_loss": [],
                        "approx_kl_div": [],
                        "clip_fraction": [], "clamp_fraction": [], "advantages": [], "max_log_prob": [],
                        "min_log_prob": [], "log_prob": [], "ratio_old_expert": [], "advantages_min": [],
                        "advantages_max": [], "ratio_old_expert_max": [], "ratio_old_expert_min": [],
                        "ratio_current_old_max": [], "ratio_current_old_min": [], "mahalanobis_distance": [], "log_prob_expert": []}

        confidence_level = 0.8

        # Calculate Chi-Square value for 95.7% confidence and n dimensions
        chi2_value = chi2.ppf(confidence_level, df=self.policy.log_std.shape[-1])  # action_dim

        # Mahalanobis distance threshold (square root of chi2). Distance from Gaussian distribution follows chi2 distribution
        mahalanobis_threshold = np.sqrt(chi2_value)
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            print("Epoch ", epoch, "of ", self.n_epochs)
            approx_kl_divs = []
            offline_data_buffer = self.expert_buffer.get(self.batch_size//2)
            online_data_buffer = self.rollout_buffer.get(self.batch_size//2)
            # Do a complete pass on the rollout buffer
            while True:
                try:
                    # Get offline batch from buffer
                    offline_batch = next(offline_data_buffer)
                    # Get online batch from buffer
                    online_batch = next(online_data_buffer)
                except StopIteration:
                    break

                actions_offline = offline_batch.actions
                actions_online = online_batch.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions_online = online_batch.actions.long().flatten()
                    actions_offline = offline_batch.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Evaluate online actions
                values_online, log_prob_online, entropy_online = self.policy.evaluate_actions(
                    online_batch.observations,
                    actions_online,
                )

                # Evaluate offline actions
                values_offline, log_prob_offline, entropy_offline, = self.policy.evaluate_actions(
                    offline_batch.observations,
                    actions_offline,
                )



                #Predict actions and log_probs for offline actions
                with th.no_grad():
                    # actions_online_expert = self.policy._predict(offline_batch.observations, deterministic=True)
                    # diff = actions_online_expert - actions_offline
                    # mahalanobis_distance_squared = ((diff ** 2) / (self.policy.log_std ** 2+1e-8)).sum(dim=-1)
                    # mahalanobis_distance = th.sqrt(mahalanobis_distance_squared)
                    #Calculate mahalanobis distance using log_std and log_prob

                    n = self.policy.log_std.shape[-1]  # action_dim

                    # If std is same across dimensions:
                    log_std_sum = torch.sum(self.policy.log_std, dim=-1)  # sum over dimensions

                    # Compute Mahalanobis squared
                    D_M_squared = -2 * (log_prob_offline + 0.5 * n * th.log(th.tensor(2*th.pi)) + log_std_sum)

                    # Mahalanobis distance
                    mahalanobis_distance = torch.sqrt(D_M_squared + 1e-8)  # small epsilon for numerical stability

                    # print(mahalanobis_distance)
                # min_log_prob = -1.*self.env.action_space.shape[0]

                old_log_prob_offline = offline_batch.old_log_prob
                log_prob_expert = offline_batch.log_prob_expert

                # Dial to weigh expert data vs online data. If expert data is from a similar policy (Inputs/Outputs) keep it lowg
                values_online = values_online.flatten()
                values_offline = values_offline.flatten()


                # Normalize advantage

                ratio_old_expert_offline = th.clamp(th.exp(
                    offline_batch.old_log_prob - log_prob_expert), max = 1.0, min=1e-3)
                # ratio_old_expert_offline_uc = th.clamp(th.exp(
                #     offline_batch.old_log_prob - log_prob_expert), max=1.0)

                advantages_online = online_batch.advantages

                advantages_offline = offline_batch.advantages

                # Concatenate advantages
                advantages = th.cat((advantages_online, advantages_offline), 0)
     

                # #Clamp advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # advantages = th.clamp(advantages, min=-10., max=10.)
                advantages_online = advantages[:len(advantages_online)]
                advantages_offline = advantages[len(advantages_online):]
                # Scale advantages for rare events - Gradient at most will be A*k
                # max_grad = 0.1 #Max norm is 0.5
                # min_sigma = 0.2
                # Sample weight: down-weight far/OOD offline samples
                weight_offline = th.where(
                    mahalanobis_distance > mahalanobis_threshold,
                    mahalanobis_threshold / (mahalanobis_distance + 1e-8),
                    th.ones_like(mahalanobis_distance),
                )
                weight_offline = th.clamp(weight_offline, 0.0, 1.0)


                #Clamp advantages
                # advantages_offline = th.clamp(advantages_offline, min=-1., max=1.)
                # advantages_online = th.clamp(advantages_online, min=-1., max=1.)
                log_ratio_current_old_offline = log_prob_offline - old_log_prob_offline
                # Max value before exp overflows in float64
                # max_exp_input = 709.0  for float64
                # log_ratio_current_old_offline = th.clamp(log_ratio_current_old_offline, min=-10, max=10)
                log_ratio_current_old_offline = log_prob_offline - old_log_prob_offline
                log_ratio_current_old_offline = th.clamp(log_ratio_current_old_offline, -20.0, 20.0)
                log_ratio_current_expert_offline = log_prob_offline - log_prob_expert
                log_ratio_current_expert_offline = th.clamp(log_ratio_current_expert_offline, -20.0, 20.0)
                ratio_current_old_offline = th.exp(log_ratio_current_old_offline)
                ratio_current_old_offline = th.clamp(ratio_current_old_offline, 0.2, 5.0)
                ratio_current_expert_offline = th.exp(log_ratio_current_expert_offline)
                ratio_current_expert_offline = th.clamp(ratio_current_expert_offline, 1e-3, 2.0)

                ratio_current_old_online = th.exp(log_prob_online - online_batch.old_log_prob)
                # ratio_current_old_offline = th.exp(
                #     log_ratio_current_old_offline)
                # ratio_current_old_offline = th.clamp(ratio_current_old_offline, 0.2, 5.0)
                # ratio_current_expert_offline = th.exp(
                #     log_prob_offline - log_prob_expert)
                # ratio_current_expert_offline = th.clamp(ratio_current_expert_offline, max=1.0, min=1e-3)

                # clipped surrogate loss for online

                policy_loss_1_online = advantages_online * ratio_current_old_online
                policy_loss_2_online = advantages_online * th.clamp(ratio_current_old_online, 1 - clip_range,
                                                                    1 + clip_range)
                policy_loss_online = -th.mean(th.min(policy_loss_1_online, policy_loss_2_online))

                # clipped surrogate loss for offline
                #normalizs the ratio_current_old_offline
                policy_loss_1_offline = advantages_offline * ratio_current_old_offline * ratio_current_expert_offline
                # Old/expert ratio is already multiplied
                # old/expert*current/old = current/expert (Expert sampled the data)
                policy_loss_2_offline = advantages_offline * th.clamp(ratio_current_old_offline, 1 - clip_range,
                                                                     1 + clip_range)* ratio_current_expert_offline
                policy_loss_offline = -th.mean(th.min(policy_loss_1_offline, policy_loss_2_offline)* weight_offline)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred_online = values_online
                    values_pred_offline = values_offline
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred_online = online_batch.old_values + th.clamp(
                        values_online - online_batch.old_values, -clip_range_vf, clip_range_vf
                    )
                    values_pred_offline = offline_batch.old_values + th.clamp(
                        values_offline - offline_batch.old_values, -clip_range_vf, clip_range_vf
                    )

                # New
                clamped_returns_offline = offline_batch.returns#th.clamp(offline_batch.returns, min=-100, max=1000)
                clamped_returns_online = online_batch.returns#th.clamp(online_batch.returns, min=-100, max=1000)
                # Value loss using the TD(gae_lambda) target
                value_loss_online = th.mean(
                    (((clamped_returns_online - values_pred_online) ** 2))) * self.vf_coef

                #Only increase value, do not decrease it. As in the future when support loses the values goes to 0

                # value_diff = (clamped_returns_offline - values_pred_offline)
                # value_loss_offline = th.mean(
                #     ((value_diff ** 2))) * self.vf_coef

                if entropy_online is None:
                    # Approximate entropy when no analytical form
                    entropy_loss_online = -th.mean(-log_prob_online + 1e-8) * self.ent_coef
                else:
                    entropy_loss_online = -th.mean(entropy_online) * self.ent_coef

                if entropy_offline is None:
                    # Approximate entropy when no analytical form
                    entropy_loss_offline = -th.mean(-log_prob_offline + 1e-8) * self.ent_coef
                else:
                    entropy_loss_offline = -th.mean(entropy_offline) * self.ent_coef

                online_loss = policy_loss_online + entropy_loss_online + value_loss_online
                offline_loss = policy_loss_offline

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    # Approx KL Divergence -- Try to keep them vv low (For online+BC this value below 0.02 works)
                    log_ratio_online = log_prob_online - online_batch.old_log_prob
                    approx_kl_div_online = th.mean(
                        ((th.exp(log_ratio_online) - 1) - log_ratio_online)).cpu().numpy()
                    log_ratio_offline = log_prob_offline - old_log_prob_offline
                    # print(log_prob_offline, old_log_prob_offline, th.exp(log_ratio_offline), advantages_offline)
                    # print("Epoch ", epoch)
                    # print("New log prob ", log_prob_offline)
                    # print("Old log prob ", old_log_prob_offline)
                    # print("Ratio ", th.exp(log_ratio_offline))
                    # print("Advantages ", advantages_offline)
                    # print("Ratio current old offline ", ratio_current_old_offline)
                    # print("Policy loss 1 ", policy_loss_1_offline)
                    # print("Policy loss 2 ", policy_loss_2_offline)
                    # print("Policy loss offline ", policy_loss_offline)
                    # print("Mahalanobis distance ", mahalanobis_distance)
                    approx_kl_div_offline = th.mean(
                        ((th.exp(log_ratio_offline) - 1) - log_ratio_offline)).cpu().numpy()

                    # Adjust learning rate to keep clip at about 0.2 - 0.3
                clip_fraction_online = th.mean(
                    (th.abs(ratio_current_old_online - 1) > clip_range).float()).item()
                clip_fraction_offline = th.mean(
                    (th.abs(ratio_current_old_offline - 1) > clip_range).float()).item()
                clamp_fraction_offline = th.mean((mahalanobis_distance > mahalanobis_threshold).float()).item()

                # Log online data
                logs_online["ratio_current_old"].append(ratio_current_old_online.mean().item())
                logs_online["policy_loss"].append(policy_loss_online.item())
                logs_online["value_loss"].append(value_loss_online.item())
                logs_online["entropy_loss"].append(entropy_loss_online.item())
                logs_online["approx_kl_div"].append(approx_kl_div_online)
                logs_online["clip_fraction"].append(clip_fraction_online)
                logs_online["advantages"].append(advantages_online.mean().item())
                logs_online["log_prob"].append(log_prob_online.mean().item())
                logs_online["advantages"].append(advantages_online.mean().item())

                # Log offline data
                logs_offline["ratio_current_old"].append(ratio_current_old_offline.mean().item())
                logs_offline["ratio_current_old_max"].append(ratio_current_old_offline.max().item())
                logs_offline["ratio_current_old_min"].append(ratio_current_old_offline.min().item())
                logs_offline["log_prob_expert"].append(log_prob_expert.mean().item())
                logs_offline["policy_loss"].append(policy_loss_offline.item())
                # logs_offline["value_loss"].append(value_loss_offline.item())
                logs_offline["entropy_loss"].append(entropy_loss_offline.item())
                logs_offline["approx_kl_div"].append(approx_kl_div_offline)
                logs_offline["clip_fraction"].append(clip_fraction_offline)
                logs_offline["clamp_fraction"].append(clamp_fraction_offline)
                logs_offline["advantages"].append(advantages_offline.mean().item())
                logs_offline["max_log_prob"].append(log_prob_offline.max().item())
                logs_offline["min_log_prob"].append(log_prob_offline.min().item())
                logs_offline["log_prob"].append(log_prob_offline.mean().item())
                logs_offline["ratio_old_expert"].append(ratio_old_expert_offline.mean().item())
                logs_offline["ratio_old_expert_max"].append(ratio_old_expert_offline.max().item())
                logs_offline["ratio_old_expert_min"].append(ratio_old_expert_offline.min().item())
                logs_offline["advantages_min"].append(advantages_offline.min().item())
                logs_offline["advantages_max"].append(advantages_offline.max().item())
                logs_offline["mahalanobis_distance"].append(mahalanobis_distance.mean().item())
                # print(advantages_offline.min().item(), advantages_offline.max().item(), ratio_old_expert_offline.mean().item(),)
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss_offline = offline_loss / 2
                loss_offline.backward()
                # # Log std should only be updated for online data
                self.policy.log_std.grad.zero_()
                loss_online = online_loss / 2
                loss_online.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                with th.no_grad():
                    self.policy.log_std.clamp_(min = -1.6)

        self._n_updates += 1
 # Logs
        # Log online data
        explained_var_online = explained_variance(self.rollout_buffer.values.flatten(),
                                                  self.rollout_buffer.returns.flatten())
        explained_var_offline = explained_variance(self.expert_buffer.values.flatten(),
                                                   self.expert_buffer.returns.flatten())
        # Logs
        # Log online data
        # Log online data
        for key, value in logs_online.items():
            self.logger.record("train_online/{}".format(key), safe_mean(value))
        # Log offline data
        for key, value in logs_offline.items():
            self.logger.record("train_offline/{}".format(key), safe_mean(value))

        self.logger.record("train_online/explained_variance", explained_var_online)
        self.logger.record("train_offline/explained_variance_offline", explained_var_offline)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.log_from_rollout_buffer(self.expert_buffer, 'train_offline/')
        self.log_from_rollout_buffer(self.rollout_buffer, 'train_online/')

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def log_from_rollout_buffer(self, buffer, prefix):
        self.logger.record(prefix + "returns", np.mean(buffer.returns))
        self.logger.record(prefix + "values", np.mean(buffer.values))
        self.logger.record(prefix + "advantages_buffer", np.mean(buffer.advantages))
        self.logger.record(prefix + "explained_variance",
                           explained_variance(buffer.values.flatten(), buffer.returns.flatten()))


    def train_bc(
        self,
        total_epochs: int,
        bc_batch_size: Optional[int] = None,
        bc_save_path: Optional[str] = None,
        log_interval: int = 100,
        bc_coeff: float = 1.0,
        warm_start_steps: int = 0,
        val_dataloader: Optional[DataLoader] = None,
        val_batches_per_epoch: Optional[int] = None,
        save_every_epochs: int = 5,
    ) -> None:
        """Train the policy on the offline dataset via behavioral cloning."""

        if self.minari_transition_dataset is None or self.minari_transition_dataloader is None:
            raise ValueError("BC training requires a Minari dataset and dataloader.")

        batch_size = bc_batch_size or self.batch_size
        transitions_per_batch = max(1, batch_size * self.num_expert_envs)
        total_transitions = len(self.minari_transition_dataset)
        print(f"Total transitions in Minari dataset: {total_transitions}")
        print(f"Total batches per epoch: {math.ceil(total_transitions / transitions_per_batch)}")
        batches_per_epoch = max(1, math.ceil(total_transitions / transitions_per_batch))

        if val_dataloader is not None:
            val_batches_per_epoch = val_batches_per_epoch or batches_per_epoch

        iterator = self.minari_transition_iterator
        if iterator is None:
            iterator = iter(self.minari_transition_dataloader)
        self.minari_transition_iterator = iterator

        self.policy.set_training_mode(True)
        bc_losses: list[float] = []
        steps = 0
        for epoch in range(total_epochs):
            iterator = iter(self.minari_transition_dataloader)
            self.minari_transition_iterator = iterator
            pbar = tqdm(total=batches_per_epoch, desc=f"BC epoch {epoch + 1}/{total_epochs}", leave=False)
            for batch_idx in range(batches_per_epoch):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break

                obs = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                obs = obs.reshape(-1, *obs.shape[2:])
                actions = actions.reshape(-1, *actions.shape[2:])

                if isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, spaces.MultiDiscrete):
                    actions = actions.long()

                _, log_prob, _ = self.policy.evaluate_actions(obs, actions)
                bc_loss = -bc_coeff * th.mean(log_prob)

                self.policy.optimizer.zero_grad()
                bc_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                bc_losses.append(bc_loss.item())
                steps += 1

                if getattr(self, "_logger", None) is not None:
                    self.logger.record("train_bc/loss", bc_loss.item())
                if log_interval > 0 and steps % log_interval == 0:
                    avg_loss = sum(bc_losses[-log_interval:]) / min(len(bc_losses), log_interval)
                    # print(f"BC epoch {epoch + 1} batch {batch_idx + 1} loss {avg_loss:.6f}")

                # Lightweight progress indicator
                pbar.update(1)
                pbar.set_postfix(loss=bc_loss.item())

            pbar.close()

            # Validation loop
            if val_dataloader is not None:
                val_iterator = iter(val_dataloader)
                val_losses: list[float] = []
                for _ in range(val_batches_per_epoch or 1):
                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        break

                    obs_v = val_batch["observations"].to(self.device)
                    acts_v = val_batch["actions"].to(self.device)
                    obs_v = obs_v.reshape(-1, *obs_v.shape[2:])
                    acts_v = acts_v.reshape(-1, *acts_v.shape[2:])
                    if isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, spaces.MultiDiscrete):
                        acts_v = acts_v.long()

                    with th.no_grad():
                        _, log_prob_v, _ = self.policy.evaluate_actions(obs_v, acts_v)
                        val_losses.append((-th.mean(log_prob_v) * bc_coeff).item())

                if val_losses:
                    val_mean = sum(val_losses) / len(val_losses)
                    print(f"Epoch {epoch+1}: val_bc_loss={val_mean:.6f}")
                    if getattr(self, "_logger", None) is not None:
                        self.logger.record("val_bc/loss", val_mean)

            # Save only after 50% of epochs have completed
            min_save_epoch = 0#max(1, math.ceil(total_epochs * 0.5))
            if (
                bc_save_path is not None
                and save_every_epochs > 0
                and (epoch + 1) >= min_save_epoch
                and (epoch + 1) % save_every_epochs == 0
            ):
                # Optional value-only finetune on fresh rollouts before saving
                if warm_start_steps > 0:
                    self._value_finetune_rollouts(num_episodes=100)
                save_path = Path(bc_save_path+f"_epoch{epoch+1} with avg loss {avg_loss:.4f}")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                self.policy.save(save_path.as_posix())
                print(f"Saved BC policy to {save_path}.zip")

    def _value_finetune_rollouts(self, num_episodes: int = 100) -> None:
        """Collect on-policy rollouts and update only the value function head."""

        if self.env is None:
            return

        self.policy.set_training_mode(True)
        episodes_done = 0
        while episodes_done < num_episodes:
            obs = self.env.reset()
            # Ensure float32 for MPS compatibility
            if isinstance(obs, np.ndarray):
                obs = obs.astype(np.float32)
            done = False
            traj_obs = []
            traj_rewards = []

            while not done:
                # Convert to float32 for MPS compatibility
                obs_tensor = th.as_tensor(np.array(obs, dtype=np.float32), device=self.device)
                with th.no_grad():
                    # Pass obs as numpy array with float32 dtype
                    obs_for_predict = obs if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32)
                    action, _ = self.policy.predict(obs_for_predict, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                if isinstance(obs, np.ndarray):
                    obs = obs.astype(np.float32)
                traj_obs.append(obs_tensor)
                traj_rewards.append(reward)

            # Compute discounted returns for the trajectory
            returns = []
            g = 0.0
            for r in reversed(traj_rewards):
                g = float(r) + self.gamma * g
                returns.insert(0, g)

            obs_batch = th.cat([o if o.ndim > 1 else o.unsqueeze(0) for o in traj_obs]).to(self.device)
            returns_batch = th.tensor(returns, dtype=th.float32, device=self.device)

            values = self.policy.predict_values(obs_batch).squeeze(-1)
            value_loss = F.mse_loss(values, returns_batch) * self.vf_coef
            # print(f"Value finetune episode {episodes_done+1}/{num_episodes}, loss: {value_loss.item():.6f}")

            self.policy.optimizer.zero_grad()
            value_loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            episodes_done += 1

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        else:
            clip_range_vf = None
        self.train_ppo_offline(clip_range, clip_range_vf)

    def _excluded_save_params(self) -> list[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. Extends parent's list to exclude unhashable objects
        like DataLoaders and iterators which are recreated on load.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        excluded_params = super()._excluded_save_params()
        # Add minari-specific objects that are unhashable and should be excluded
        excluded_params.extend([
            "minari_transition_dataloader",
            "minari_transition_iterator", 
            "minari_transition_dataset",
        ])
        return excluded_params

    def learn(
            self: SelfPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,

            log_interval: int = 1,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        updated = False

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)
            continue_training = self.make_offline_rollouts(callback, self.expert_buffer,
                                                           n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)


            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

            # if self._current_progress_remaining < 0.5 and not updated:
            #     with th.no_grad():
            #         std = th.exp(self.policy.log_std)
            #         std += 0.5
            #         self.policy.log_std.copy_(th.log(std))
            #
            #     updated = True

        callback.on_training_end()

        return self
