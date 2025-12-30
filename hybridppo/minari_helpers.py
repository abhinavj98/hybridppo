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

class MinariTransitionDataset(Dataset):
    def __init__(self, minari_dataset):
        self.minari_dataset = minari_dataset
        self.index_map = []

        #Index - (episode_id, step_id) for each transition
        for ep_id, episode in enumerate(minari_dataset):
            for step in range(len(episode.actions) - 1):
                self.index_map.append((ep_id, step))

        # (episode_id, step_id) -> index mapping
        self.episode_to_indices = {}
        for idx, (ep_id, step_id) in enumerate(self.index_map):
            self.episode_to_indices.setdefault(ep_id, []).append(idx)

    def __getitem__(self, idx):
        # Get the episode and step from the index map and return
        # Custom sampling logic makes sure that the index are in order

        ep_id, step = self.index_map[idx]
        episode = self.minari_dataset[ep_id]

        return {
            "episode_ids": torch.tensor(ep_id),
            "step_ids": torch.tensor(step),
            "observations": torch.tensor(episode.observations[step], dtype=torch.float32),
            "actions": torch.tensor(episode.actions[step], dtype=torch.float32),
            "rewards": torch.tensor(episode.rewards[step], dtype=torch.float32),
            "next_observations": torch.tensor(episode.observations[step + 1], dtype=torch.float32),
            "dones": torch.tensor(
                episode.terminations[step] or episode.truncations[step],
                dtype=torch.bool
            )
        }

    def __len__(self):
        return len(self.index_map)

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

        self.ep_indices = dataset.episode_to_indices
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