from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, ReplayBufferSamples, DictReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize

from hybridppo.minari_helpers import get_dataset, MinariTransitionDataset
from hybridppo.rlpd_policy import RLPDPolicy

class RLPD(SAC):
    def __init__(
        self,
        policy: Union[str, Type[RLPDPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[Any] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        minari_dataset: Union[str, Any] = None,
        minari_dataset_kwargs: Optional[Dict[str, Any]] = None,
        mix_ratio: float = 0.5,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.minari_dataset_ref = minari_dataset
        self.minari_dataset_kwargs = minari_dataset_kwargs or {}
        self.mix_ratio = mix_ratio
        self.offline_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Create offline buffer
        # We assume n_envs=1 for offline buffer as it's just a collection of transitions
        # We need to make sure the buffer size is large enough to hold the dataset
        # Or we can just use the dataset length.

        # Load dataset
        if isinstance(self.minari_dataset_ref, str):
             # Assume format "env_name" or "dataset_name"
             # If it's a path or name, we might need more info.
             # get_dataset(dataset: str, env: str, names) signature
             # We need to parse or assume.
             # But here we accept object or string.
             # If string, we assume it is the 'names' argument and we guess 'dataset' and 'env'?
             # Or we expect the user to pass the dataset object.
             # For simplicity, if it's a string, we warn or fail if we can't infer.
             # Ideally, the user passes the object or we use the kwargs.
             pass

        # If minari_dataset_ref is not None, we load it.
        if self.minari_dataset_ref is not None:
            # Check if it's already a MinariTransitionDataset
            if isinstance(self.minari_dataset_ref, MinariTransitionDataset):
                self.minari_dataset = self.minari_dataset_ref
            else:
                # Assume it's a MinariDataset object (from minari.load_dataset) or we need to wrap it
                # If it's a string, we probably can't load it easily without env name.
                # We assume the user passed a MinariDataset object or similar.
                self.minari_dataset = MinariTransitionDataset(self.minari_dataset_ref, preload=True)

            n_transitions = len(self.minari_dataset)

            # Create buffer
            # We use the same class as online buffer
            self.offline_buffer = self.replay_buffer_class(
                buffer_size=n_transitions,
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                n_envs=1, # Offline data treated as 1 env
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

            print(f"Populating offline buffer with {n_transitions} transitions...")

            # Bulk load if preloaded
            if self.minari_dataset.preload:
                # Convert tensors to numpy
                obs = self.minari_dataset.observations.cpu().numpy()
                next_obs = self.minari_dataset.next_observations.cpu().numpy()
                actions = self.minari_dataset.actions.cpu().numpy()
                rewards = self.minari_dataset.rewards.cpu().numpy()
                dones = self.minari_dataset.dones.cpu().numpy()

                # Check shapes. Buffer expects (size, n_envs, dim)
                # Data is (size, dim). We expand dims.
                obs = obs[:, None, :]
                next_obs = next_obs[:, None, :]
                actions = actions[:, None, :]
                rewards = rewards[:, None] # (size, 1)
                dones = dones[:, None]

                # Directly assign to buffer arrays (unsafe but fast)
                # Verify sizes match
                # self.offline_buffer.observations[:n_transitions] = obs
                # ...
                # Actually, ReplayBuffer implementation details might vary (DictReplayBuffer etc)
                # But self.replay_buffer_class is likely ReplayBuffer or DictReplayBuffer.

                if isinstance(self.offline_buffer, ReplayBuffer):
                     self.offline_buffer.observations = obs
                     self.offline_buffer.next_observations = next_obs
                     self.offline_buffer.actions = actions
                     self.offline_buffer.rewards = rewards
                     self.offline_buffer.dones = dones
                     self.offline_buffer.timeouts = np.zeros_like(dones) # Assuming no timeouts or not tracked
                     self.offline_buffer.pos = n_transitions if n_transitions < self.offline_buffer.buffer_size else 0 # Full
                     self.offline_buffer.full = True # We set size=n_transitions so it is full
                else:
                    # Fallback to loop for DictReplayBuffer or others
                    for i in range(n_transitions):
                        self.offline_buffer.add(
                            self.minari_dataset[i]['observations'].cpu().numpy(),
                            self.minari_dataset[i]['next_observations'].cpu().numpy(),
                            self.minari_dataset[i]['actions'].cpu().numpy(),
                            self.minari_dataset[i]['rewards'].cpu().numpy(),
                            self.minari_dataset[i]['dones'].cpu().numpy(),
                            [{}], # infos
                        )
            else:
                 # Loop load
                 for i in range(n_transitions):
                        data = self.minari_dataset[i]
                        self.offline_buffer.add(
                            data['observations'].cpu().numpy(),
                            data['next_observations'].cpu().numpy(),
                            data['actions'].cpu().numpy(),
                            data['rewards'].cpu().numpy(),
                            data['dones'].cpu().numpy(),
                            [{}], # infos
                        )

            print("Offline buffer populated.")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer

            # Determine split
            offline_batch_size = int(batch_size * self.mix_ratio)
            online_batch_size = batch_size - offline_batch_size

            # Get online samples
            # Check if we have enough online data
            if self.replay_buffer.size() < online_batch_size:
                 # If not enough online data, use what we have or fill with offline?
                 # Standard behavior: wait for learning_starts.
                 # If we are here, learning_starts should be met.
                 # But self.replay_buffer.size() counts stored transitions.
                 # If batch_size > current size, SB3 might error or repeat.
                 # We assume learning_starts >= batch_size.
                 pass

            online_data = self.replay_buffer.sample(online_batch_size, env=self._vec_normalize_env)

            # Get offline samples
            if self.offline_buffer is not None and offline_batch_size > 0:
                offline_data = self.offline_buffer.sample(offline_batch_size, env=self._vec_normalize_env)

                # Concatenate
                # ReplayBufferSamples fields: observations, actions, next_observations, dones, rewards
                # They are tensors.

                replay_data = ReplayBufferSamples(
                    observations=th.cat([online_data.observations, offline_data.observations]),
                    actions=th.cat([online_data.actions, offline_data.actions]),
                    next_observations=th.cat([online_data.next_observations, offline_data.next_observations]),
                    dones=th.cat([online_data.dones, offline_data.dones]),
                    rewards=th.cat([online_data.rewards, offline_data.rewards]),
                )
            else:
                replay_data = online_data

            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = self.gamma # Assuming 1-step for now as defaults
            # SB3 SAC doesn't support n-step by default in buffer?
            # In updated SB3, replay_data might have discounts field if using HerReplayBuffer or similar?
            # Standard ReplayBufferSamples usually doesn't have discounts?
            # Let's check the source I read:
            # discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            # ReplayBufferSamples might not have discounts attribute in older versions, but current SB3 likely has it or uses subclass.
            # I will use safe access.
            if hasattr(replay_data, "discounts") and replay_data.discounts is not None:
                discounts = replay_data.discounts
            else:
                discounts = self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
