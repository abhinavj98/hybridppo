
import gymnasium as gym
import minari
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from argparse import ArgumentParser
from hybridppo.minari_helpers import get_dataset, get_environment, get_eval_environment

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="D4RL")
parser.add_argument("--env", type=str, default="door")
parser.add_argument("--name", type=str, default="human")

args = parser.parse_args()
dataset_name = f"{args.dataset}/{args.env}/{args.name}"

dataset = get_dataset(args.dataset, args.env, args.name)
print(dataset.env_spec)
env = get_environment(dataset)
eval_env = get_eval_environment(dataset, render_mode="human")

model = PPO.load("ppo_model", device="cpu")

for i in range(2):
    obs, info = eval_env.reset()
    terminated = False
    truncated = False
    while not terminated or not truncated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        eval_env.render()
