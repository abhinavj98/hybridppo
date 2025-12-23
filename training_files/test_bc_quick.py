"""Minimal smoke test for BC training/validation pipeline using local dataset."""
import os
import sys
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial
import gymnasium as gym
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybridppo.ppo_expert import PPOExpert
from hybridppo.policies import MlpPolicyExpert
from hybridppo.minari_helpers import get_dataset, MultiEpisodeSequentialSampler, collate_env_batch


class SubsetTransitionDataset(Dataset):
    """Subset that yields step-level transitions for the sampler."""
    def __init__(self, dataset, episode_ids):
        self.dataset = dataset
        self.episode_ids = list(episode_ids)
        self.index_map = []
        for ep_id in self.episode_ids:
            episode = self.dataset[ep_id]
            for step in range(len(episode.actions) - 1):
                self.index_map.append((ep_id, step))
        
        # Build episode_to_indices for sampler
        self.episode_to_indices = {}
        for idx, (ep_id, step_id) in enumerate(self.index_map):
            self.episode_to_indices.setdefault(ep_id, []).append(idx)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ep_id, step = self.index_map[idx]
        episode = self.dataset[ep_id]
        obs = episode.observations[step]
        next_obs = episode.observations[step + 1]
        return {
            "episode_ids": torch.tensor(ep_id),
            "step_ids": torch.tensor(step),
            "observations": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(episode.actions[step], dtype=torch.float32),
            "rewards": torch.tensor(episode.rewards[step], dtype=torch.float32),
            "next_observations": torch.tensor(next_obs, dtype=torch.float32),
            "dones": torch.tensor(
                episode.terminations[step] or episode.truncations[step],
                dtype=torch.bool,
            ),
        }


def build_loader(dataset, n_envs=1, batch_size=2):
    sampler = MultiEpisodeSequentialSampler(dataset, n_envs=n_envs, batch_size=batch_size)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(collate_env_batch, n_envs=n_envs, batch_size=batch_size),
        num_workers=0,
        shuffle=False,
    )


def main():
    # Load actual Minari dataset
    dataset = get_dataset("mujoco", "walker2d", ["expert-v0"])
    if dataset is None:
        raise ValueError("Dataset not found")
    
    env = gym.make(dataset.env_spec)
    
    # Use just first 3 episodes for quick test (2 train, 1 val)
    num_episodes = min(3, len(dataset))
    train_ids = list(range(num_episodes - 1))  # [0, 1]
    val_ids = [num_episodes - 1]  # [2]
    
    train_ds = SubsetTransitionDataset(dataset, train_ids)
    val_ds = SubsetTransitionDataset(dataset, val_ids)
    
    n_envs = 1
    n_steps = 32
    train_loader = build_loader(train_ds, n_envs=n_envs, batch_size=n_steps)
    val_loader = build_loader(val_ds, n_envs=n_envs, batch_size=n_steps)
    
    policy_kwargs = {
        "log_std_init": 0.0,
        "activation_fn": nn.ReLU,
    }
    
    model = PPOExpert(
        MlpPolicyExpert,
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=32,
        n_epochs=1,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        device="cpu",
        minari_dataset=dataset,
        log_prob_expert=0,
    )

    model.minari_transition_dataset = train_ds
    model.minari_transition_dataloader = train_loader
    model.minari_transition_iterator = iter(train_loader)

    print(f"Running BC on {len(train_ds)} train transitions, {len(val_ds)} val transitions")
    model.train_bc(
        total_epochs=5,
        bc_batch_size=32,
        bc_save_path="test",
        log_interval=1,
        bc_coeff=1.0,
        warm_start_steps=1,
        val_dataloader=val_loader,
        val_batches_per_epoch=1,
        save_every_epochs=1,
    )
    env.close()
    print("BC smoke test completed successfully!")


if __name__ == "__main__":
    main()
