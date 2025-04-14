# Load dataset and environment
# Run PPO using stable baselines3
# Visualize results

import gymnasium as gym
import minari
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from argparse import ArgumentParser
from hybridppo.minari_helpers import get_dataset, get_environment, get_eval_environment
from torch import nn
import yaml
from stable_baselines3.common.env_util import make_vec_env
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D4RL")
    parser.add_argument("--env", type=str, default="door")
    parser.add_argument("--name", type=str, default="human")
    parser.add_argument("--hparam", type=str, default="human")
    parser.add_argument("--save_file", type=str, default="human")
    parser.add_argument("--num_runs", type=int, default=1)
    args = parser.parse_args()
    dataset_name = f"{args.dataset}/{args.env}/{args.name}"
    # path = "C:/Users/abhin/OneDrive/Desktop/hybrid-ppo/"

    with open("hparam.yml", "r") as f:
        hparam_all = yaml.safe_load(f)
    hparam = hparam_all[args.hparam]
    dataset = get_dataset(args.dataset, args.env, args.name)
    print(dataset.env_spec)
    print(hparam)

    for i in range(args.num_runs):
        env = make_vec_env(lambda: gym.make(dataset.env_spec), n_envs=hparam['n_envs'], monitor_dir=None)
        # Run PPO
        policy_kwargs = {}  # {"net_arch": {"pi": [64,64], "vf": [64, 64]}, "activation_fn": nn.Tanh}
        model = PPO(hparam['policy'], env, verbose=1, learning_rate=hparam['learning_rate'], n_steps=hparam['n_steps'],
                          batch_size=hparam['batch_size'], n_epochs=hparam['n_epochs'],
                          gamma=hparam['gamma'], ent_coef=hparam['ent_coef'], clip_range=hparam['clip_range'],
                          normalize_advantage=hparam['normalize'], vf_coef=hparam['vf_coef'],
                          gae_lambda=hparam['gae_lambda'], max_grad_norm=hparam['max_grad_norm'],
                          policy_kwargs=policy_kwargs, tensorboard_log = './tb_test/online/'+dataset_name, device='cpu',)

        print(" Running on device", model.device)
        model.learn(total_timesteps=hparam['n_timesteps'])
        # Save model
        model.save(args.save_file+f"_{i}")
        #Close environment
        env.close()
