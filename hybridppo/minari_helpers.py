# Helper functions for minari
import minari
import gymnasium as gym
from hybridppo import DATASET_PATH
from torch.utils.data import IterableDataset, get_worker_info
from minari.dataset.episode_data import EpisodeData
from torch.utils.data import Sampler
import random
from torch.utils.data import Sampler
import random

def get_dataset(dataset: str, env: str, names) -> minari.MinariDataset:
    # names can be a single string or an iterable of strings
    if isinstance(names, str):
        names = [names]
    temp_name = "_".join(names)
    print(temp_name)
    name = f"{DATASET_PATH}/{env}/{temp_name}"
    # try:
    #     dataset = minari.load_dataset(name)
    #     return dataset
    # except FileNotFoundError:
    #     pass
    
    # Try to load individual datasets from local path first
    datasets = []
    for i in names:
        local_path = f"{DATASET_PATH}/{dataset}/{env}/{i}"
        try:
            datasets.append(minari.load_dataset(local_path))
        except FileNotFoundError:
            # If not found locally, try downloading
            url = f"{dataset}/{env}/{i}"
            downloaded = minari.download_dataset(url, force_download=True)
            #Save to local path
            print(f"Downloading dataset from {url} to {local_path}")
            new_dataset = minari.load_dataset(url)
            print(f"downloaded dataset has {len(new_dataset)} episodes")
            datasets.append(new_dataset)

    print(f"Loaded {len(datasets)} datasets for names {names}")
    
    if len(datasets) == 1:
        return datasets[0]
    final_dataset = minari.combine_datasets(datasets, name)
    print(f"Combined dataset has {len(final_dataset)} episodes")
    return final_dataset

def get_environment(minari_dataset: minari.MinariDataset, **kwargs) -> gym.Env:
    return minari_dataset.recover_environment(**kwargs)

def get_eval_environment(minari_dataset: minari.MinariDataset, **kwargs) -> gym.Env:
    return minari_dataset.recover_environment(eval_env=True, **kwargs)



from torch.utils.data import Dataset
import torch

import numpy as np

class MinariTransitionDataset(Dataset):
    def __init__(self, minari_dataset, preload=True):
        self.minari_dataset = minari_dataset
        self.preload = preload
        
        # Build episode index map (episode_id, step_in_episode) -> global_idx
        self.index_map = []
        total_transitions = 0
        
        for ep_id, episode in enumerate(minari_dataset):
            if len(episode.actions) < 1:
                continue
            n_steps = len(episode.actions)
            for step_id in range(n_steps):
                self.index_map.append((ep_id, step_id))
            total_transitions += n_steps
        
        if preload:
            print(f"Pre-loading {len(minari_dataset)} episodes into RAM...")
            self._preload_data()
        else:
            print(f"Lazy loading enabled for {total_transitions} transitions from {len(minari_dataset)} episodes.")
    
    def _preload_data(self):
        obs_list = []
        act_list = []
        rew_list = []
        next_obs_list = []
        done_list = []
        ep_id_list = []
        step_id_list = []

        for ep_id, episode in enumerate(self.minari_dataset):
            # Skip empty episodes
            if len(episode.actions) < 1:
                continue

            # Correct indexing (observations has length T+1, everything else has length T):
            # obs[i]: observations[i] for i in [0, T-1] -> observations[:-1]
            # next_obs[i]: observations[i+1] for i in [0, T-1] -> observations[1:]
            # actions/rewards/terminations/truncations: all length T, use all elements
            
            obs_list.append(episode.observations[:-1])
            next_obs_list.append(episode.observations[1:])
            act_list.append(episode.actions)
            rew_list.append(episode.rewards)
            
            terminations = episode.terminations
            truncations = episode.truncations
            dones = np.logical_or(terminations, truncations).astype(np.float32)
            done_list.append(dones)
            
            # Metadata
            n_steps = len(episode.actions)
            ep_id_list.append(np.full(n_steps, ep_id, dtype=np.int64))
            step_id_list.append(np.arange(n_steps, dtype=np.int64))

        # Concatenate all into single tensors
        self.observations = torch.tensor(np.concatenate(obs_list), dtype=torch.float32)
        self.next_observations = torch.tensor(np.concatenate(next_obs_list), dtype=torch.float32)
        self.actions = torch.tensor(np.concatenate(act_list), dtype=torch.float32)
        self.rewards = torch.tensor(np.concatenate(rew_list), dtype=torch.float32)
        self.dones = torch.tensor(np.concatenate(done_list), dtype=torch.float32)
        self.episode_ids_tensor = torch.tensor(np.concatenate(ep_id_list), dtype=torch.int64)
        self.step_ids_tensor = torch.tensor(np.concatenate(step_id_list), dtype=torch.int64)
        
        print(f"Loaded {len(self.observations)} transitions.")

    def __len__(self):
        if self.preload:
            return len(self.observations)
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.preload:
            return {
                'observations': self.observations[idx],
                'actions': self.actions[idx],
                'rewards': self.rewards[idx],
                'next_observations': self.next_observations[idx],
                'dones': self.dones[idx],
                'episode_ids': self.episode_ids_tensor[idx],
                'step_ids': self.step_ids_tensor[idx],
            }
        else:
            # Lazy loading
            ep_id, step_id = self.index_map[idx]
            episode = self.minari_dataset[ep_id]
            
            obs = episode.observations[step_id]
            next_obs = episode.observations[step_id + 1]
            action = episode.actions[step_id]
            reward = episode.rewards[step_id]
            
            termination = episode.terminations[step_id]
            truncation = episode.truncations[step_id]
            done = float(termination or truncation)
            
            return {
                'observations': torch.tensor(obs, dtype=torch.float32),
                'actions': torch.tensor(action, dtype=torch.float32),
                'rewards': torch.tensor(reward, dtype=torch.float32),
                'next_observations': torch.tensor(next_obs, dtype=torch.float32),
                'dones': torch.tensor(done, dtype=torch.float32),
                'episode_ids': torch.tensor(ep_id, dtype=torch.int64),
                'step_ids': torch.tensor(step_id, dtype=torch.int64),
            }
    
    def get_episode_to_indices(self):
        """Build episode_to_indices mapping for sampler compatibility."""
        episode_to_indices = {}
        current_idx = 0
        
        for ep_id, episode in enumerate(self.minari_dataset):
            if len(episode.actions) < 1:
                continue
            n_steps = len(episode.actions)
            episode_to_indices[ep_id] = list(range(current_idx, current_idx + n_steps))
            current_idx += n_steps
        
        return episode_to_indices

    def __len__(self):
        if self.preload:
            return len(self.observations)
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.preload:
            return {
                'observations': self.observations[idx],
                'actions': self.actions[idx],
                'rewards': self.rewards[idx],
                'next_observations': self.next_observations[idx],
                'dones': self.dones[idx],
                'episode_ids': self.episode_ids_tensor[idx],
                'step_ids': self.step_ids_tensor[idx],
            }
        else:
            # Lazy loading
            ep_id, step_id = self.index_map[idx]
            episode = self.minari_dataset[ep_id]
            
            obs = episode.observations[step_id]
            next_obs = episode.observations[step_id + 1]
            action = episode.actions[step_id]
            reward = episode.rewards[step_id]
            
            termination = episode.terminations[step_id]
            truncation = episode.truncations[step_id]
            done = float(termination or truncation)
            
            return {
                'observations': torch.tensor(obs, dtype=torch.float32),
                'actions': torch.tensor(action, dtype=torch.float32),
                'rewards': torch.tensor(reward, dtype=torch.float32),
                'next_observations': torch.tensor(next_obs, dtype=torch.float32),
                'dones': torch.tensor(done, dtype=torch.float32),
                'episode_ids': torch.tensor(ep_id, dtype=torch.int64),
                'step_ids': torch.tensor(step_id, dtype=torch.int64),
            }


class MultiEpisodeSequentialSampler(Sampler):
    """The purpose of this sample is to yield indexes of transitions
    from multiple episodes in a sequential manner. Returned indexes are
    flattened as [e0t0, e0t1, e0t2, ..., e0tn, e1t0, e1t1, ..., e(num_envs-1)t(batch_size-1)]
    so the list size is num_envs * batch_size. Use collate_fn to reorder"""
    def __init__(self, dataset, n_envs, batch_size, seed=None):
        super().__init__(dataset)
        self.dataset = dataset
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        self.ep_indices = dataset.get_episode_to_indices()
        self.available_episodes = list(self.ep_indices.keys())

    def _sample_episode(self):
        return self.rng.choice(self.available_episodes)

    def __iter__(self):

        #Sample n_envs episodes
        env_its = {
            env_id: iter(self.ep_indices[self._sample_episode()])
            for env_id in range(self.n_envs)
        }

        while True:
            batch = []
            for env_id in range(self.n_envs):
                ep_batch = []
                while len(ep_batch) < self.batch_size:
                    ep_iter = env_its[env_id]
                    try:
                        ep_batch.append(next(ep_iter)) #For episode go through corresponding indices
                    except StopIteration:
                        # Episode ended, resample another one
                        new_ep = self._sample_episode()
                        env_its[env_id] = iter(self.ep_indices[new_ep]) #Update iterator to new episode once exhausted

                batch.extend(ep_batch)

            yield batch

    def __len__(self):
        return float("inf")  # Infinite stream


def collate_env_batch(batch, n_envs, batch_size):
    assert len(batch) == n_envs * batch_size, (
        f"Expected batch of size {n_envs * batch_size}, got {len(batch)}"
    )

    grouped = [batch[i * batch_size:(i + 1) * batch_size] for i in range(n_envs)] #batch, n_envs
    collated = {}
    #Shape as batch_size, n_envs, ...
    for key in batch[0].keys(): #For each key in the batch
        batch_stacked = [torch.stack([b[key] for b in group]) for group in grouped] #batch, envs, ...
        batch_stacked = torch.stack(batch_stacked, dim=1)
        collated[key] = batch_stacked
    return collated  # Shape: [batch_size, n_envs, ...]