import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import torch
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybridppo.rlpd import RLPD
from hybridppo.rlpd_policy import RLPDPolicy
from hybridppo.minari_helpers import get_dataset
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mujoco")
    parser.add_argument("--env", type=str, default="Walker2d-v4")
    parser.add_argument("--names", nargs='+', required=True)
    parser.add_argument("--hparam", type=str, default="Walker2d-v4-rlpd")
    parser.add_argument("--save_dir", type=str, default="rlpd_checkpoints")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Torch device for RLPD training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_timesteps", type=int, default=None, help="Overwrite hparam n_timesteps")

    args = parser.parse_args()

    # Load hparams
    with open("hparam.yml", "r") as f:
        hparam_all = yaml.safe_load(f)
    hparam = hparam_all.get(args.hparam, hparam_all.get("default"))
    if hparam is None:
        raise ValueError(f"Could not find hparam {args.hparam}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        torch.cuda.manual_seed_all(args.seed)

    # Load Dataset
    dataset = get_dataset(args.dataset, args.env, args.names)
    if dataset is None:
        raise ValueError("Dataset not found")

    # Environment setup
    n_envs = hparam.get("n_envs", 1)
    env_id = dataset.env_spec.id if hasattr(dataset, 'env_spec') else args.env

    # We use DummyVecEnv for simplicity, but can use SubprocVecEnv if n_envs > 1
    # RLPD often uses n_envs=1
    env = DummyVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])

    if hparam.get("normalize", True):
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Setup RLPD
    policy_kwargs = {}
    # If we want to pass net_arch
    # policy_kwargs["net_arch"] = [256, 256]

    # Device setup
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

    model = RLPD(
        policy=RLPDPolicy,
        env=env,
        learning_rate=hparam.get("learning_rate", 3e-4),
        buffer_size=hparam.get("buffer_size", 1_000_000),
        learning_starts=hparam.get("learning_starts", 5000),
        batch_size=hparam.get("batch_size", 256),
        tau=hparam.get("tau", 0.005),
        gamma=hparam.get("gamma", 0.99),
        train_freq=hparam.get("train_freq", 1),
        gradient_steps=hparam.get("gradient_steps", 1),
        ent_coef=hparam.get("ent_coef", "auto"),
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"logs/rlpd/{args.env}",
        verbose=1,
        seed=args.seed,
        device=device,
        minari_dataset=dataset,
        mix_ratio=hparam.get("mix_ratio", 0.5),
    )

    save_root = Path(args.save_dir) / args.dataset / args.env / ''.join(args.names)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = args.save_name or f"rlpd_{args.env}_{timestamp}"
    save_path = save_root / save_name

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(save_root),
        name_prefix=save_name,
    )

    n_timesteps = args.n_timesteps or int(hparam.get("n_timesteps", 1e6))

    print(f"Starting training for {n_timesteps} timesteps...")
    model.learn(total_timesteps=n_timesteps, callback=checkpoint_callback, log_interval=10)

    model.save(str(save_path))
    print(f"Model saved to {save_path}")
    env.close()
