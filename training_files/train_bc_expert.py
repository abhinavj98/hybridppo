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
    """A subset of MinariTransitionDataset filtered by episode ids, pre-loaded into RAM."""

    def __init__(self, minari_dataset, episode_ids):
        self.minari_dataset = minari_dataset
        self.episode_ids = list(episode_ids)
        
        obs_list = []
        act_list = []
        rew_list = []
        next_obs_list = []
        done_list = []
        ep_id_list = []
        step_id_list = []

        print(f"Pre-loading {len(self.episode_ids)} episodes into RAM...")
        for ep_id in self.episode_ids:
            episode = self.minari_dataset[ep_id]
            
            # Skip empty episodes
            if len(episode.actions) < 1:
                continue

            # We use the same slicing logic as the original code:
            # It iterates range(len(episode.actions) - 1), effectively skipping the last transition.
            # Assuming len(obs) == len(act) + 1
            
            # obs: 0 to T-2 (inclusive) -> slice [:-2]
            # next_obs: 1 to T-1 (inclusive) -> slice [1:-1]
            # actions/rewards/dones: 0 to T-2 -> slice [:-1]
            
            obs_list.append(episode.observations[:-2])
            next_obs_list.append(episode.observations[1:-1])
            act_list.append(episode.actions[:-1])
            rew_list.append(episode.rewards[:-1])
            
            terminations = episode.terminations[:-1]
            truncations = episode.truncations[:-1]
            dones = np.logical_or(terminations, truncations).astype(np.float32)
            done_list.append(dones)
            
            # Metadata
            n_steps = len(episode.actions) - 1
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

    def __getitem__(self, idx):
        return {
            "episode_ids": self.episode_ids_tensor[idx],
            "step_ids": self.step_ids_tensor[idx],
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones": self.dones[idx],
        }

    def __len__(self):
        return len(self.observations)

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
    # parser.add_argument("--warm_start_steps", type=int, default=0, help="If >0, run value-only finetune rollouts each epoch")
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Save BC checkpoint every N epochs (<=0 to disable)")
    parser.add_argument("--save_dir", type=str, default="bc_checkpoints")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of episodes for validation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Torch device for BC training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
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
    print(f"DEBUG: torch.cuda.is_available() -> {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"DEBUG: torch.version.cuda -> {torch.version.cuda}")
        print(f"DEBUG: torch.cuda.device_count() -> {torch.cuda.device_count()}")
        print(f"DEBUG: torch.cuda.current_device() -> {torch.cuda.current_device()}")
        print(f"DEBUG: torch.cuda.get_device_name(0) -> {torch.cuda.get_device_name(0)}")


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

    train_loader = DataLoader(
        train_ds,
        batch_size=transitions_per_batch,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=transitions_per_batch,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"DEBUG: Calling PPOExpert with device={device}")
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
        val_dataloader=val_loader,
        val_batches_per_epoch=val_batches_per_epoch,
        save_every_epochs=args.save_every_epochs,
        metrics_path=metrics_path,
    )

    env.close()
