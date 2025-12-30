# Load dataset and environment
# Run PPO using stable baselines3
# Visualize results
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import minari
from copy import deepcopy
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
#Import linear LR, Scheduler
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import yaml
import random
import string
import wandb
from stable_baselines3.common.callbacks import EvalCallback
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
    # parser.add_argument("--name", type=str, default="human")
    parser.add_argument("--hparam", type=str, default="human")
    # parser.add_argument("--log_prob_expert", type=int, default=0)
    parser.add_argument("--save_file", type=str, default="human")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument('--names', nargs='+', help='List of names', required=True)
    parser.add_argument('--r', type=float, default=0.001, help='Ratio for log_prob_expert')
    parser.add_argument('--bc_policy', type=str, default=None, help='Path to a pretrained BC checkpoint to initialize the policy')
    parser.add_argument('--mix_ratio', type=float, default=0.5, help='Offline/online minibatch split ratio (0..1)')
    parser.add_argument('--rho_bar', type=float, default=1.0, help='V-trace rho_bar cap')
    parser.add_argument('--c_bar', type=float, default=0.95, help='V-trace c_bar cap')
    parser.add_argument('--log_std_subtract', type=float, default=0.0, help='Subtract this constant from log_std after each update')
    parser.add_argument("--seed", type=int, default=42)

    
    args = parser.parse_args()
    dataset_name = f"{args.dataset}/{args.env}/{args.names}"
    # path = "C:/Users/abhin/OneDrive/Desktop/hybrid-ppo/"
    random.seed(args.seed)
    np.random.seed(args.seed)


    with open("hparam.yml", "r") as f:
        hparam_all = yaml.safe_load(f)
    hparam = hparam_all[args.hparam]
    dataset = get_dataset(args.dataset, args.env, args.names)
    if dataset is None:
        raise ValueError("Dataset not found")
    print(dataset.env_spec)
    print(hparam)

 

    # env = get_environment(dataset)
    # # Wrap the environment with DummyVecEnv
    # env = make
    # eval_env = get_eval_environment(dataset, render_mode="human")
    # #Print device
    r = args.r
    print(r)
    for i in range(args.num_runs):
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        wandb_params = {'dataset_name': dataset_name,
                        # 'log_prob_expert': args.log_prob_expert,
                        'name': dataset_name+random_string+args.save_file+str(i),
                        'r': r,
                        }
        wandb_params.update(hparam)
        wandb_params['mix_ratio'] = float(max(0.0, min(1.0, args.mix_ratio)))
        wandb_params['rho_bar'] = args.rho_bar
        wandb_params['c_bar'] = args.c_bar
        wandb_params['log_std_subtract'] = max(0.0, args.log_std_subtract)
        if args.bc_policy:
            wandb_params['bc_policy'] = args.bc_policy

        init_wandb(wandb_params)
        env = make_vec_env(lambda: gym.make(dataset.env_spec), n_envs=hparam['n_envs'], monitor_dir=None)
        eval_env = gym.make(dataset.env_spec, render_mode="human")
        eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                     log_path="./logs/", eval_freq=50000,
                                     deterministic=True, render=False)
        #Eval callback

        log_prob_expert = r#-np.log(r) + env.action_space.shape[0] * 0.699 #-logr -D/2logpi
        # Run PPO
        policy_kwargs = {"log_std_init": hparam['log_std_init'],
                         "activation_fn": nn.ReLU, "optimizer_kwargs": {"betas": (0.999, 0.999)}, }
        # {"net_arch": {"pi": [64,64], "vf": [64, 64]}, "activation_fn": nn.Tanh}
        model = PPOExpert(MlpPolicyExpert, env, verbose=1, learning_rate=hparam['learning_rate'], n_steps=hparam['n_steps'],
                          batch_size=hparam['batch_size'], n_epochs=hparam['n_epochs'],
                          gamma=hparam['gamma'], ent_coef=hparam['ent_coef'], clip_range=hparam['clip_range'],
                          normalize_advantage=hparam['normalize'], vf_coef=hparam['vf_coef'],
                          gae_lambda=hparam['gae_lambda'], max_grad_norm=hparam['max_grad_norm'],
                  policy_kwargs = policy_kwargs, tensorboard_log = './tb_test/hybrid/'+dataset_name+'/'+args.save_file+str(i), device = 'cpu',
                  minari_dataset = dataset,log_prob_expert=log_prob_expert,
                  mix_ratio=wandb_params['mix_ratio'], rho_bar=wandb_params['rho_bar'], c_bar=wandb_params['c_bar'],
                  log_std_subtract=wandb_params['log_std_subtract'],
                          )
        print(" Running on device", model.device)
        if args.bc_policy:
            if not os.path.isfile(args.bc_policy):
                raise FileNotFoundError(f"Specified BC policy not found: {args.bc_policy}")
            bc_policy = MlpPolicyExpert.load(args.bc_policy, device=model.device)
            model.policy.load_state_dict(bc_policy.state_dict())
            model.expert_policy = deepcopy(model.policy)
            print(f"Loaded BC policy weights from {args.bc_policy}")
        model.learn(total_timesteps=hparam['n_timesteps'], callback=eval_callback)

        # Evaluate model
        # mean_reward, std_reward = evaluate_policwany(model, eval_env, n_eval_episodes=10)
        # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save model
        try:
            model.save(args.save_file+f"_{i}", exclude=['minari_transition_iterator'])
        except Exception as e:
            print("Error saving model: ", e)
        wandb.finish()

        # Visualize results by running 5 episodes
