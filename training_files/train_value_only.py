import sys
import os
from copy import deepcopy
import gymnasium as gym
import torch as th
import torch.nn as nn
import yaml
from argparse import ArgumentParser
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybridppo.policies import MlpPolicyExpert
from hybridppo.ppo_expert import PPOExpert

# Usage example:
# python training_files/train_value_only.py --env Walker2d-v4 --model_path bc_checkpoints/mujoco/walker2d/expert-v0_epoch50.zip --timesteps 100000

class PPOValueOnly(PPO):
    """
    PPO subclass that ignores minari_dataset argument and can be used to load PPOBC checkpoints
    for value-only training.
    """
    def __init__(self, policy, env, minari_dataset=None, **kwargs):
        # Remove minari_dataset from kwargs if present to avoid error in PPO.__init__
        # (though we accepted it as explicit arg, so it won't be in kwargs unless passed twice)
        super().__init__(policy, env, **kwargs)

def freeze_actor(model):
    """
    Freezes the actor network and action head.
    """
    # Freeze policy network (actor)
    # In MlpPolicyExpert (and ActorCriticPolicy), the actor is in mlp_extractor.policy_net
    # and the action head is action_net.
    
    policy = model.policy
    
    # Freeze mlp_extractor.policy_net
    if hasattr(policy, "mlp_extractor"):
        # Check if policy_net exists (MlpExtractor)
        if hasattr(policy.mlp_extractor, "policy_net"):
            for param in policy.mlp_extractor.policy_net.parameters():
                param.requires_grad = False
        # If it's a shared extractor (e.g. CNN), we might not want to freeze it if value net uses it.
        # But MlpPolicyExpert usually has separate or split.
        
    # Freeze action_net
    if hasattr(policy, "action_net"):
        for param in policy.action_net.parameters():
            param.requires_grad = False
            
    # Freeze log_std if it exists (for diagonal gaussian)
    if hasattr(policy, "log_std"):
        policy.log_std.requires_grad = False

    print("Actor network frozen.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, help="Environment ID (e.g. Walker2d-v4)")
    parser.add_argument("--dataset", type=str, help="Minari dataset name")
    parser.add_argument("--minari_env", type=str, help="Minari env name")
    parser.add_argument("--names", nargs='+', help="Minari dataset names")
    parser.add_argument("--hparam", type=str, default=None, help="Hparam key (e.g. Walker2d-v4-bc-large)")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained BC policy (.zip)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to train value function")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for value function training")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the tuned model")
    
    args = parser.parse_args()
    
    # Resolve environment ID
    dataset = None
    if args.dataset:
        from hybridppo.minari_helpers import get_dataset
        print(f"Resolving environment from dataset {args.dataset}/{args.minari_env}/{args.names}...")
        dataset = get_dataset(args.dataset, args.minari_env, args.names)
        if dataset is None:
            raise ValueError("Could not load dataset to resolve environment ID")
        # Minari dataset.env_spec might be an EnvSpec object or similar, we need the id
        if hasattr(dataset, 'env_spec') and hasattr(dataset.env_spec, 'id'):
             env_id = dataset.env_spec.id
        elif hasattr(dataset, 'env_spec'): # Sometimes it's just the spec
             env_id = dataset.env_spec.id if hasattr(dataset.env_spec, 'id') else str(dataset.env_spec)
        else:
             # Fallback or error
             raise ValueError("Could not determine env_id from dataset.env_spec")
        print(f"Resolved environment ID: {env_id}")
    else:
        raise ValueError("Must provide either --env or (--dataset, --minari_env, --names)")

    # Create environment
    # We use make_vec_env for parallel environments if needed, or just one.
    # Let's use 1 env for simplicity or check hparams if we want to be fancy.
    # But for value training, more envs = faster rollout. 
    # Load hyperparameters
    hparam = {}
    if args.hparam:
        hparam_path = os.path.join(os.path.dirname(__file__), '..', 'hparam.yml')
        if os.path.exists(hparam_path):
            with open(hparam_path, "r") as f:
                hparam_all = yaml.safe_load(f)
            hparam = hparam_all.get(args.hparam, hparam_all.get("default", {}))
            print(f"Loaded hparams for {args.hparam}")

    # Create environment
    n_envs = args.n_envs if args.n_envs > 1 else hparam.get('n_envs', 1)
    print(f"Creating {n_envs} environments for {env_id}")
    env = make_vec_env(env_id, n_envs=n_envs)
    
   
    # Determine device
    device = args.device
    print(f"Using device: {device}")

    # Initialize model
    print(f"Initializing PPOExpert model with hparams...")

    # Prepare policy kwargs
    policy_kwargs = {
        "log_std_init": hparam.get("log_std_init", 0.0),
        "activation_fn": nn.ReLU, 
        "optimizer_kwargs": {"betas": (0.999, 0.999)}
    }
    
    # Default params for PPOExpert specific args
    r = 0.001
    log_prob_expert = r
    
    model = PPOValueOnly(
        MlpPolicyExpert,
        env,
        verbose=1,
        learning_rate=args.learning_rate if args.learning_rate is not None else hparam.get("learning_rate", 3e-4),
        n_steps=hparam.get("n_steps", 512),
        batch_size=hparam.get("batch_size", 64),
        n_epochs=hparam.get("n_epochs", 20),
        gamma=hparam.get("gamma", 0.99),
        ent_coef=hparam.get("ent_coef", 0.000585),
        clip_range=hparam.get("clip_range", 0.1),
        normalize_advantage=hparam.get("normalize", True),
        vf_coef=hparam.get("vf_coef", 0.87),
        gae_lambda=hparam.get("gae_lambda", 0.95),
        max_grad_norm=hparam.get("max_grad_norm", 1.0),
        policy_kwargs=policy_kwargs,
        device=device,
        minari_dataset=dataset
    )
    
    print(f"Loading BC policy weights from {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
        
    bc_policy = MlpPolicyExpert.load(args.model_path, device=device)
    model.policy.load_state_dict(bc_policy.state_dict())
    model.expert_policy = deepcopy(model.policy)
    print("Successfully loaded policy weights into PPOExpert model.")

    # Freeze actor
    freeze_actor(model)
    
    # Enable gradients for value function
    # In MlpPolicyExpert, the value function is in mlp_extractor.value_net and value_net
    if hasattr(model.policy, "mlp_extractor"):
        if hasattr(model.policy.mlp_extractor, "value_net"):
            for param in model.policy.mlp_extractor.value_net.parameters():
                param.requires_grad = True
    if hasattr(model.policy, "value_net"):
        for param in model.policy.value_net.parameters():
            param.requires_grad = True
            
    # Train
    print(f"Training value function for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    
    # Save
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = args.model_path.replace(".zip", "_value_tuned.zip")
        
    model.policy.save(save_path)
    print(f"Saved value-tuned policy to {save_path}")
