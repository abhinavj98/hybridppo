import argparse
import time
from typing import Sequence

import numpy as np

from hybridppo.minari_helpers import get_dataset, get_environment


def replay_dataset(
    dataset: str,
    env: str,
    names: Sequence[str],
    max_episodes: int | None = None,
    render: bool = False,
    sleep: float = 0.0,
) -> None:
    """Replay episodes from a Minari dataset in the original environment.

    This is mainly for sanity-checking that the dataset matches the env
    (rewards, terminations, truncations) and for quick visual inspection.
    """
    # Load combined Minari dataset (uses local path or downloads via helper)
    minari_dataset = get_dataset(dataset, env, names)
    print(f"Loaded dataset with {len(minari_dataset)} episodes")

    # Recover the original environment used to collect the dataset
    env_instance = get_environment(minari_dataset, render_mode="human" if render else None)
    print(f"Recovered environment: {env_instance}")

    num_episodes = len(minari_dataset) if max_episodes is None else min(len(minari_dataset), max_episodes)

    for ep_idx in range(num_episodes):
        episode = minari_dataset[ep_idx]
        dataset_return = float(np.sum(episode.rewards))

        obs, info = env_instance.reset()
        ep_return_env = 0.0
        steps = 0

        # for action, reward, terminated, truncated in zip(
        #     episode.actions,
        #     episode.rewards,
        #     episode.terminations,
        #     episode.truncations,
        # ):
        #     obs, env_reward, env_terminated, env_truncated, info = env_instance.step(action,)
        #     ep_return_env += float(env_reward)
        #     steps += 1

        #     if render:
        #         env_instance.render()
        #     if sleep > 0.0:
        #         time.sleep(sleep)

        #     if env_terminated or env_truncated:
        #         break

        print(
            f"Episode {ep_idx}: steps={steps}, "
            f"dataset_return={dataset_return:.3f}, env_return={ep_return_env:.3f}"
            f"Epsiode length: {len(episode.actions)}"
        )

    env_instance.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Minari dataset in its original environment")
    parser.add_argument("--dataset", type=str, required=True, help="Top-level dataset group (e.g., mujoco, D4RL)")
    parser.add_argument("--env", type=str, required=True, help="Environment key used in get_dataset (e.g., walker2d)")
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset names to load and (optionally) combine",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Max number of episodes to replay")
    parser.add_argument("--render", action="store_true", help="Render the environment during replay")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep (in seconds) between environment steps when rendering",
    )

    args = parser.parse_args()

    replay_dataset(
        dataset=args.dataset,
        env=args.env,
        names=args.names,
        max_episodes=args.episodes,
        render=args.render,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
