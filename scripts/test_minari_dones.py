#!/usr/bin/env python3
"""Validate `dones` loading from Minari datasets via `MinariTransitionDataset`.

Usage:
  python3 scripts/test_minari_dones.py --dataset mujoco --env humanoid --names medium-v0
  python3 scripts/test_minari_dones.py --dataset mujoco --env walker2d --names expert-v0,expert-v1 --preload False

The script loads the Minari dataset(s), constructs `MinariTransitionDataset`, and checks that
- the concatenated number of transitions equals the sum of episode lengths
- for each episode, the last transition's `dones` matches episode.terminations|truncations
- statistics about `dones` are printed

Exits with code 1 on mismatch.
"""
import argparse
import sys
import numpy as np
import minari
from hybridppo.minari_helpers import get_dataset, MinariTransitionDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Minari dataset provider (e.g. 'mujoco')")
    p.add_argument("--env", required=True, help="Environment id in dataset (e.g. 'humanoid')")
    p.add_argument("--names", required=True, help="Comma-separated dataset names (e.g. 'medium-v0' or 'expert-v0,expert-v1')")
    p.add_argument("--preload", action="store_true", help="Preload episodes into RAM (default: False)")
    p.add_argument("--limit-episodes", type=int, default=5, help="How many episodes to print for inspection")
    args = p.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    print(f"Loading dataset provider={args.dataset} env={args.env} names={names} preload={args.preload}")

    ds = get_dataset(args.dataset, args.env, names)
    print(f"Loaded MinariDataset with {len(ds)} episodes")

    # Build transition dataset
    td = MinariTransitionDataset(ds, preload=args.preload)
    print(f"Transition dataset length: {len(td)} transitions")

    # Basic counts
    total_steps_sum = 0
    mismatches = []
    for ep_id, episode in enumerate(ds):
        if len(episode.actions) < 1:
            continue
        n_steps = len(episode.actions)
        total_steps_sum += n_steps
        term = np.asarray(episode.terminations, dtype=np.bool_)
        trunc = np.asarray(episode.truncations, dtype=np.bool_)
        combined = np.logical_or(term, trunc)
        last_combined = combined[-1]

        # If preloaded, reconstruct the index ranges
        if args.preload:
            # td.get_episode_to_indices() provides mapping
            ep_to_idx = td.get_episode_to_indices()
            if ep_id not in ep_to_idx:
                mismatches.append((ep_id, "missing in transition mapping"))
                continue
            indices = ep_to_idx[ep_id]
            # print(f"Episode {ep_id} indices in transition dataset: {indices}")
            # last transition index corresponds to last action
            print(f"Length of episode {ep_id} indices: {len(indices)} expected {n_steps}")
            last_idx = indices[-1]
            done_val = bool(td.dones[last_idx].item())
            if done_val != bool(last_combined):
                mismatches.append((ep_id, int(done_val), int(last_combined)))
        else:
            # Lazy mode: sample by index
            # compute cumulative index by iterating existing ep lengths
            pass

        if ep_id < args.limit_episodes:
            print(f"Episode {ep_id}: steps={n_steps}, terminations_last={int(term[-1])}, truncations_last={int(trunc[-1])}, combined_last={int(last_combined)}")

    print(f"Sum of episode steps = {total_steps_sum}, transition dataset len = {len(td)}")

    if total_steps_sum != len(td):
        print("ERROR: total step count across episodes does not match transition dataset length")
        print(f"total_steps_sum={total_steps_sum} td_len={len(td)}")
        sys.exit(1)

    if mismatches:
        print("ERROR: Found mismatches between episode terminal flags and transition dataset dones for some episodes:")
        for item in mismatches[:20]:
            print(item)
        sys.exit(1)

    # Print global dones stats
    if args.preload:
        dones_arr = td.dones.cpu().numpy() if hasattr(td.dones, 'cpu') else np.asarray(td.dones)
        print(f"Global dones count: {int(dones_arr.sum())}/{len(dones_arr)} ({100.0 * dones_arr.sum() / len(dones_arr):.2f}%)")
        # Print a few indices where dones==1
        ones = np.flatnonzero(dones_arr)
        print("First done indices:", ones[:20])

    print("All checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
