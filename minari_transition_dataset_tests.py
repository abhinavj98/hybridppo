import minari
import torch
from torch.utils.data import DataLoader
from hybridppo.minari_helpers import MinariTransitionDataset, get_dataset, MultiEpisodeSequentialSampler, collate_env_batch
from argparse import ArgumentParser
from functools import partial
import time
# def test_transition_dataloader(dataset_id: str, args: ArgumentParser,  batch_size: int = 129):
#     # Load Minari dataset
#     print(f"Loading Minari dataset: {dataset_id}")
#     minari_dataset = get_dataset(args.dataset, args.env, args.name)
#
#     # Wrap in transition dataset
#     transition_dataset = MinariTransitionDataset(minari_dataset)
#     print(f"Total transitions: {len(transition_dataset)}")
#
#     # Wrap in DataLoader
#     dataloader = DataLoader(transition_dataset, batch_size=batch_size, shuffle=True)
#
#     # Grab one batch and inspect
#     batch = next(iter(dataloader))
#     print(f"obs shape:       {batch['observations'].shape}")
#     print(f"action shape:    {batch['actions'].shape}")
#     print(f"reward shape:    {batch['rewards'].shape}")
#     print(f"next_obs shape:  {batch['next_observations'].shape}")
#     print(f"done shape:      {batch['dones'].shape}")
#
#     # Print one example
#     print("\n First sample in batch:")
#     print("obs:", batch["observations"][0])
#     print("action:", batch["actions"][0])
#     print("reward:", batch["rewards"][0].item())
#     print("next_obs:", batch["next_observations"][0])
#     print("done:", batch["dones"][0].item())

def test_transition_dataloader(dataset_id: str, args: ArgumentParser, n_envs: int = 4):
    print(f"Loading Minari dataset: {dataset_id}")
    minari_dataset = get_dataset(args.dataset, args.env, args.name)
    transition_dataset = MinariTransitionDataset(minari_dataset)
    for num_worksers in [1, 2, 4, 8, 16]:
        parallel_sequential_sampler = MultiEpisodeSequentialSampler(
            transition_dataset,
            n_envs=n_envs,
            batch_size=128,
        )
        start_time = time.time()
        dataloader = DataLoader(
            transition_dataset,
            batch_sampler=parallel_sequential_sampler,
            collate_fn=partial(collate_env_batch, n_envs=n_envs, batch_size=128),
            num_workers=num_worksers
        )
        dataloader_iter = iter(dataloader)
        print(f"Total transitions: {len(transition_dataset)}")
        # print(f"Total batches: {len(dataloader)}")
        for i in range(50):
            batch = next(dataloader_iter)
            # print(f"Batch {i+1}:")
            # print(f"obs shape:       {batch['observations'].shape}")
            # print(f"action shape:    {batch['actions'].shape}")
            # print(f"reward shape:    {batch['rewards'].shape}")
            # print(f"next_obs shape:  {batch['next_observations'].shape}")
            # print(f"done shape:      {batch['dones'].shape}")
            # print("\n First sample in batch:")
            # # print("obs:", batch["observations"][0])
            # # print("action:", batch["actions"][0])
            # # print("reward:", batch["rewards"][0].item())
            # # print("next_obs:", batch["next_observations"][0])
            # # print("done:", batch["dones"][0].item())
            # for j in range(4):
            #     print("ep, step", list(zip(batch["episode_ids"][j], batch["step_ids"][j]))[0])
        print(f"Time taken for {num_worksers} workers: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D4RL")
    parser.add_argument("--env", type=str, default="door")
    parser.add_argument("--name", type=str, default="human")

    args = parser.parse_args()
    dataset_name = f"{args.dataset}/{args.env}/{args.name}"
    test_transition_dataloader(dataset_name, args, n_envs=4)
