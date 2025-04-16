# Load dataset and environment
# Run PPO using stable baselines3
# Visualize results
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import minari
from hybridppo.ppo_expert import PPOExpert
from hybridppo.policies import MlpPolicyExpert
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from argparse import ArgumentParser
from hybridppo.minari_helpers import get_dataset, get_environment, get_eval_environment
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from hybridppo.policies import MlpPolicyExpert
from stable_baselines3.common.env_util import make_vec_env

import yaml
import wandb
def init_wandb(params):
    wandb.init(
        project="HybridPPO",
        entity="pruning-rl",
        config=params,
        name=params['name'],
        sync_tensorboard=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D4RL")
    parser.add_argument("--env", type=str, default="door")
    parser.add_argument("--name", type=str, default="human")
    parser.add_argument("--hparam", type=str, default="human")
    parser.add_argument("--log_prob_expert", type=int, default=0)
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

    wandb_params = {'dataset_name': dataset_name,
                    'log_prob_expert': args.log_prob_expert,
                    'name': args.save_file,
                    }
    wandb_params.update(hparam)


    init_wandb(wandb_params)

    # env = get_environment(dataset)
    # # Wrap the environment with DummyVecEnv
    # env = make
    # eval_env = get_eval_environment(dataset, render_mode="human")
    # #Print device

    for i in range(args.num_runs):
        env = make_vec_env(lambda: gym.make(dataset.env_spec), n_envs=hparam['n_envs'], monitor_dir=None)
        # Run PPO
        policy_kwargs = {"log_std_init": hparam['log_std_init'],
                         "activation_fn": nn.ReLU, "optimizer_kwargs": {"betas": (0.999, 0.999)},
                         }
        # {"net_arch": {"pi": [64,64], "vf": [64, 64]}, "activation_fn": nn.Tanh}
        model = PPOExpert(MlpPolicyExpert, env, verbose=1, learning_rate=hparam['learning_rate'], n_steps=hparam['n_steps'],
                          batch_size=hparam['batch_size'], n_epochs=hparam['n_epochs'],
                          gamma=hparam['gamma'], ent_coef=hparam['ent_coef'], clip_range=hparam['clip_range'],
                          normalize_advantage=hparam['normalize'], vf_coef=hparam['vf_coef'],
                          gae_lambda=hparam['gae_lambda'], max_grad_norm=hparam['max_grad_norm'],
                          policy_kwargs = policy_kwargs, tensorboard_log = './tb_test/hybrid/'+dataset_name+'/'+args.save_file, device = 'cpu',
                          minari_dataset = dataset,log_prob_expert=args.log_prob_expert,
                          )
        print(" Running on device", model.device)
        model.learn(total_timesteps=hparam['n_timesteps'])

        # Evaluate model
        # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save model
        model.save(args.save_file+f"_{i}", exclude=['minari_transition_iterator'])

        # Visualize results by running 5 episodes
