# Train a behavioral cloning policy using the PPOExpert data pipeline
import os
import sys
import math
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import torch
import torch.nn as nn
import yaml

from hybridppo.ppo_expert import PPOExpert
from hybridppo.policies import MlpPolicyExpert
from hybridppo.minari_helpers import get_dataset, MinariTransitionDataset, MultiEpisodeSequentialSampler, collate_env_batch
from torch.utils.data import DataLoader, Dataset


class SubsetMinariTransitionDataset(Dataset):
    """A subset of MinariTransitionDataset filtered by episode ids."""

    def __init__(self, minari_dataset, episode_ids):
        self.minari_dataset = minari_dataset
        self.episode_ids = list(episode_ids)
        self.index_map = []
        for ep_id in self.episode_ids:
            episode = self.minari_dataset[ep_id]
            for step in range(len(episode.actions) - 1):
                self.index_map.append((ep_id, step))

        self.episode_to_indices = {}
        for idx, (ep_id, step_id) in enumerate(self.index_map):
            self.episode_to_indices.setdefault(ep_id, []).append(idx)

    def __getitem__(self, idx):
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
                dtype=torch.bool,
            ),
        }

    def __len__(self):
        return len(self.index_map)

# conda activate hybrid-ppo && python ./training_files/train_bc_expert.py --dataset mujoco --env walker2d --names expert-v0 --hparam Walker2d-v4-bc-large --bc_epochs 50 --bc_batch_size 128 --bc_coeff 0.005 --warm_start_steps 100 --save_dir bc_checkpoints --log_interval 10
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D4RL")
    parser.add_argument("--env", type=str, default="door")
    parser.add_argument("--names", nargs='+', required=True)
    parser.add_argument("--hparam", type=str, default="Walker2d-v4-bc-large")
    parser.add_argument("--bc_epochs", type=int, default=5)
    parser.add_argument("--bc_batch_size", type=int, default=None)
    parser.add_argument("--bc_coeff", type=float, default=1.0, help="Coefficient to scale the BC log-prob loss")
    parser.add_argument("--warm_start_steps", type=int, default=0, help="If >0, run value-only finetune rollouts each epoch")
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Save BC checkpoint every N epochs (<=0 to disable)")
    parser.add_argument("--save_dir", type=str, default="bc_checkpoints")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of episodes for validation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Torch device for BC training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    with open("hparam.yml", "r") as f:
        hparam_all = yaml.safe_load(f)
    hparam = hparam_all.get(args.hparam, hparam_all.get("default"))
    if hparam is None:
        raise ValueError(f"Could not find hparam {args.hparam}")

    dataset = get_dataset(args.dataset, args.env, args.names)
    if dataset is None:
        raise ValueError("Dataset not found")

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy_kwargs = {
        "log_std_init": hparam.get("log_std_init", 0),
        "activation_fn": nn.ReLU,
        "optimizer_kwargs": {"betas": (0.999, 0.999)},
    }

    env = gym.make(dataset.env_spec)

    # Episode-level split for train/val
    num_episodes = len(dataset)
    val_episodes = max(1, int(math.ceil(num_episodes * args.val_fraction)))
    all_eps = np.random.permutation(num_episodes)
    val_ids = all_eps[:val_episodes]
    train_ids = all_eps[val_episodes:]

    train_ds = SubsetMinariTransitionDataset(dataset, train_ids)
    val_ds = SubsetMinariTransitionDataset(dataset, val_ids)

    n_envs = hparam.get("n_envs", 1)
    n_steps = hparam.get("n_steps", 512)
    bc_batch_size = args.bc_batch_size or hparam.get("batch_size", 64)
    transitions_per_batch = max(1, bc_batch_size * n_envs)
    val_batches_per_epoch = max(1, math.ceil(len(val_ds) / transitions_per_batch))

    train_sampler = MultiEpisodeSequentialSampler(train_ds, n_envs=n_envs, batch_size=n_steps)
    val_sampler = MultiEpisodeSequentialSampler(val_ds, n_envs=n_envs, batch_size=n_steps)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=partial(collate_env_batch, n_envs=n_envs, batch_size=n_steps),
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=partial(collate_env_batch, n_envs=n_envs, batch_size=n_steps),
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = PPOExpert(
        MlpPolicyExpert,
        env,
        verbose=1,
        learning_rate=hparam.get("learning_rate", 3e-4),
        n_steps=hparam.get("n_steps", 512),
        batch_size=hparam.get("batch_size", 64),
        n_epochs=hparam.get("n_epochs", 10),
        gamma=hparam.get("gamma", 0.99),
        ent_coef=hparam.get("ent_coef", 0.0),
        clip_range=hparam.get("clip_range", 0.2),
        normalize_advantage=hparam.get("normalize", False),
        vf_coef=hparam.get("vf_coef", 0.5),
        gae_lambda=hparam.get("gae_lambda", 0.95),
        max_grad_norm=hparam.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        device=device,
        minari_dataset=dataset,
        log_prob_expert=0,
    )

    # Override dataloaders with train/val splits
    model.minari_transition_dataset = train_ds
    model.minari_transition_dataloader = train_loader
    model.minari_transition_iterator = iter(train_loader)

    save_root = Path(args.save_dir) / args.dataset / args.env / ''.join(args.names)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = args.save_name or f"bc_{args.env}_{timestamp}"
    bc_save_path = save_root / save_name

    print(f"Training BC policy for dataset {args.names} -> saving to {bc_save_path}.zip")
    # Metrics file alongside checkpoints
    metrics_path = (bc_save_path.parent / f"{bc_save_path.name}_metrics.csv").as_posix()
    model.train_bc(
        total_epochs=args.bc_epochs,
        bc_batch_size=transitions_per_batch,  # use resolved per-update batch size
        bc_save_path=str(bc_save_path),
        log_interval=args.log_interval,
        bc_coeff=args.bc_coeff,
        warm_start_steps=args.warm_start_steps,
        val_dataloader=val_loader,
        val_batches_per_epoch=val_batches_per_epoch,
        save_every_epochs=args.save_every_epochs,
        metrics_path=metrics_path,
    )

    env.close()
