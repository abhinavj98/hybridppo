import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybridppo.minari_helpers import get_dataset

def main():
    parser = argparse.ArgumentParser(description="Compute mean and variance of returns in a Minari offline dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset group (e.g. mujoco)')
    parser.add_argument('--env', type=str, required=True, help='Environment name (e.g. walker2d)')
    parser.add_argument('--names', nargs='+', required=True, help='Dataset variant names (e.g. medium-v0)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for discounted returns (default: 0.99)')
    args = parser.parse_args()

    dataset = get_dataset(args.dataset, args.env, args.names)
    if dataset is None:
        print("Could not load dataset.")
        sys.exit(1)

    # Each episode is a dict with 'rewards' key
    returns = []
    discounted_returns = []
    gamma = args.gamma
    for ep in dataset.iterate_episodes():
        rewards = ep.rewards
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        ep_return = rewards.sum()
        # Compute discounted sum
        disc = 0.0
        for r in reversed(rewards):
            disc = r + gamma * disc
        discounted_returns.append(disc)
        returns.append(ep_return)

    returns = np.array(returns)
    discounted_returns = np.array(discounted_returns)
    print(f"Dataset: {args.dataset}/{args.env}/{','.join(args.names)}")
    print(f"Num episodes: {len(returns)}")
    print(f"Undiscounted returns:")
    print(f"  Mean: {returns.mean():.4f}")
    print(f"  Variance: {returns.var():.4f}")
    print(f"  Min: {returns.min():.4f}")
    print(f"  Max: {returns.max():.4f}")
    print(f"Discounted returns (gamma={gamma}):")
    print(f"  Mean: {discounted_returns.mean():.4f}")
    print(f"  Variance: {discounted_returns.var():.4f}")
    print(f"  Min: {discounted_returns.min():.4f}")
    print(f"  Max: {discounted_returns.max():.4f}")

if __name__ == "__main__":
    main()
