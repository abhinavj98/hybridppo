# Shows loading of a minari dataset
# Recovering Environment
# And runnning example from the dataset
import gymnasium as gym
import minari
from argparse import ArgumentParser
from hybridppo.minari_helpers import get_dataset, get_environment
parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="D4RL")
parser.add_argument("--env", type=str, default="door")
parser.add_argument("--name", type=str, default="human")

args = parser.parse_args()
dataset_name = f"{args.dataset}/{args.env}/{args.name}"
# path = "C:/Users/abhin/OneDrive/Desktop/hybrid-ppo/"

dataset = get_dataset(args.dataset, args.env, args.name)
env = get_environment(dataset, render_mode="human")

# sample 5 episodes from the dataset
episodes = dataset.sample_episodes(n_episodes=5)
#Run each episode in the environment and render
for episode in episodes:
    env.reset()
    for i in range(len(episode)):
        action = episode.actions[i]
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
env.close()