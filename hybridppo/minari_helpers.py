# Helper functions for minari
import minari
import gymnasium as gym
from hybridppo import DATASET_PATH
from torch.utils.data import IterableDataset, get_worker_info
from minari.dataset.episode_data import EpisodeData

def get_dataset(dataset: str, env: str, names: str) -> minari.MinariDataset:
    #names is a list of names
    #Create new name containing all names
    temp_name = "_".join(names)
    print(temp_name)
    name = f"{DATASET_PATH}/{dataset}/{env}/{temp_name}"
    try:
        dataset = minari.load_dataset(name)
        return dataset
    except FileNotFoundError:
        pass
    datasets = []
    for i in names:
        datasets.append(minari.load_dataset(f"{DATASET_PATH}/{dataset}/{env}/{i}"))
    return minari.combine_datasets(datasets, name)

def get_environment(minari_dataset: minari.MinariDataset, **kwargs) -> gym.Env:
    return minari_dataset.recover_environment(**kwargs)

def get_eval_environment(minari_dataset: minari.MinariDataset, **kwargs) -> gym.Env:
    return minari_dataset.recover_environment(eval_env=True, **kwargs)


# class MinariTransitionDataset(Dataset):
#     def __init__(self, minari_dataset):
#         """
#         Stream transitions (obs, action, reward, next_obs, done) from MinariDataset
#         without loading everything into memory.
#         MinariDataset is a Dataset object that returns episodes. We instead return
#         transitions (obs, action, reward, next_obs, done).
#         """
#         self.minari_dataset = minari_dataset # Minari dataset is a Dataset object that return episodes
#         self.index = []
#
#         # # Build (episode_idx, step_idx) index
#         # for episode_idx, episode in enumerate(minari_dataset):
#         #     print(episode, episode)
#         #     num_steps = len(episode)
#         #     for step_idx in range(num_steps):
#         #         self.index.append((episode_idx, step_idx))
#
#         #List comprehension with zip to make self.index
#         self.index = [(episode_idx, step_idx) for episode_idx, episode in enumerate(minari_dataset) for
#                       step_idx in range(len(episode))]
#
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, idx):
#         #Data iterator will take care of the indexing and batching
#         episode_idx, step_idx = self.index[idx]
#         episode = self.minari_dataset[episode_idx]
#
#         obs = torch.as_tensor(episode.observations[step_idx], dtype=torch.float32)
#         next_obs = torch.as_tensor(episode.observations[step_idx + 1], dtype=torch.float32)
#         action = torch.as_tensor(episode.actions[step_idx], dtype=torch.float32)
#         reward = torch.as_tensor(episode.rewards[step_idx], dtype=torch.float32)
#         done = torch.as_tensor(
#             episode.terminations[step_idx] or episode.truncations[step_idx],
#             dtype=torch.bool
#         )
#
#         return {
#             "observations": obs,
#             "actions": action,
#             "rewards": reward,
#             "next_observations": next_obs,
#             "dones": done
#         }

import random
import torch
from torch.utils.data import IterableDataset

# class MinariTransitionDataset(IterableDataset):
#     def __init__(self, minari_dataset, num_envs: int):
#         """
#         An IterableDataset that returns batches of transitions.
#         Each row in the batch comes from a separate episode and maintains temporal order.
#         When an episode ends, it is resampled from the dataset.
#         """
#         self.dataset = minari_dataset
#         self.num_envs = num_envs
#
#     def __iter__(self):
#         # For each slot, initialize an active episode and current step
#         episode_idxs = [random.randint(0, len(self.dataset) - 1) for _ in range(self.num_envs)]
#         episodes = [self.dataset[i] for i in episode_idxs]
#         step_idxs = [0] * self.num_envs
#
#         while True:
#             batch = {
#                 "observations": [],
#                 "actions": [],
#                 "rewards": [],
#                 "next_observations": [],
#                 "dones": []
#             }
#
#             for i in range(self.num_envs):
#                 episode = episodes[i]
#                 t = step_idxs[i]
#
#                 if t >= len(episode.actions):
#                     # Resample a new episode if the previous one is done
#                     new_ep_idx = random.randint(0, len(self.dataset) - 1)
#                     episode = self.dataset[new_ep_idx]
#                     episodes[i] = episode
#                     step_idxs[i] = 0
#                     t = 0
#
#                 obs = torch.tensor(episode.observations[t], dtype=torch.float32)
#                 next_obs = torch.tensor(episode.observations[t + 1], dtype=torch.float32)
#                 act = torch.tensor(episode.actions[t], dtype=torch.float32)
#                 rew = torch.tensor(episode.rewards[t], dtype=torch.float32)
#                 done = torch.tensor(
#                     episode.terminations[t] or episode.truncations[t],
#                     dtype=torch.bool
#                 )
#
#                 batch["observations"].append(obs)
#                 batch["actions"].append(act)
#                 batch["rewards"].append(rew)
#                 batch["next_observations"].append(next_obs)
#                 batch["dones"].append(done)
#
#                 step_idxs[i] += 1
#
#             yield {k: torch.stack(v) for k, v in batch.items()}

# class MinariTransitionDataset(IterableDataset):
#     """An IterableDataset that returns batches of transitions.
#     Always use with num_workers == num_online_envs.
#     As we need to simulate each trajectory being returned step by step,
#     by a separate worker, we need to make sure that each worker
#     is assigned a different episode.
#     Each worker gets passed a copy of the dataset and samples a different episode.
#     We then use worker_id to organize our data in the collate fn."""
#
#     def __init__(self, minari_dataset):
#         self.episode_dataset = minari_dataset
#         self.num_episodes = len(minari_dataset)
#
#     def __iter__(self):
#         worker_info = get_worker_info()
#         worker_id = worker_info.id if worker_info else 0
#         rng = random.Random(worker_id)  # Make worker-specific RNG
#
#         # Sample one episode for this worker
#         episode_idx = rng.randint(0, self.num_episodes - 1)
#         episode = self.episode_dataset[episode_idx]
#         step = 0
#
#         while True:
#             if step >= len(episode.actions):
#                 # Resample new episode
#                 episode_idx = rng.randint(0, self.num_episodes - 1)
#                 episode = self.episode_dataset[episode_idx]
#                 step = 0
#
#             yield {
#                 "worker_id": worker_id,
#                 "episode_id": episode_idx,
#                 "step_id": step,
#                 "observation": torch.tensor(episode.observations[step], dtype=torch.float32),
#                 "action": torch.tensor(episode.actions[step], dtype=torch.float32),
#                 "reward": torch.tensor(episode.rewards[step], dtype=torch.float32),
#                 "next_observation": torch.tensor(episode.observations[step + 1], dtype=torch.float32),
#                 "done": torch.tensor(
#                     episode.terminations[step] or episode.truncations[step],
#                     dtype=torch.bool
#                 )
#             }
#             step += 1
#
# def collate_fn(batch):
#     return batch
#     # # Sort rows by worker_id to preserve env ordering
#     # # print(batch)
#     # batch.sort(key=lambda x: x["worker_id"])
#     #
#     # return {
#     #     "worker_ids": torch.tensor([x["worker_id"] for x in batch]),
#     #     "episode_ids": torch.tensor([x["episode_id"] for x in batch]),
#     #     "step_ids": torch.tensor([x["step_id"] for x in batch]),
#     #     "observations": torch.stack([x["observation"] for x in batch]),
#     #     "actions": torch.stack([x["action"] for x in batch]),
#     #     "rewards": torch.stack([x["reward"] for x in batch]),
#     #     "next_observations": torch.stack([x["next_observation"] for x in batch]),
#     #     "dones": torch.stack([x["done"] for x in batch]),
#     # }


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


from torch.utils.data import Sampler
import random
from torch.utils.data import Sampler
import random

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