import sys
import os
from copy import deepcopy
import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math
import numpy as np
from argparse import ArgumentParser
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import obs_as_tensor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybridppo.policies import MlpPolicyExpert
from hybridppo.ppo_expert import PPOExpert


def freeze_actor(model):
    """Freeze actor network and action head to train value only."""
    policy = model.policy
    if hasattr(policy, "mlp_extractor"):
        if hasattr(policy.mlp_extractor, "policy_net"):
            for param in policy.mlp_extractor.policy_net.parameters():
                param.requires_grad = False
    if hasattr(policy, "action_net"):
        for param in policy.action_net.parameters():
            param.requires_grad = False
    if hasattr(policy, "log_std"):
        policy.log_std.requires_grad = False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, help="Environment ID (e.g. Walker2d-v4)")
    parser.add_argument("--dataset", type=str, help="Minari dataset name")
    parser.add_argument("--minari_env", type=str, help="Minari env name")
    parser.add_argument("--names", nargs='+', help="Minari dataset names")
    parser.add_argument("--hparam", type=str, default=None, help="Hparam key (e.g. Walker2d-v4-bc-large)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained BC policy (.zip)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of transitions to process from dataset")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for value updates (will default to model.batch_size)")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for value function training")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the tuned policy")
    args = parser.parse_args()

    # Resolve environment ID from dataset if provided
    dataset = None
    if args.dataset:
        from hybridppo.minari_helpers import get_dataset
        print(f"Resolving environment from dataset {args.dataset}/{args.minari_env}/{args.names}...")
        dataset = get_dataset(args.dataset, args.minari_env, args.names)
        if dataset is None:
            raise ValueError("Could not load dataset to resolve environment ID")
        if hasattr(dataset, 'env_spec') and hasattr(dataset.env_spec, 'id'):
            env_id = dataset.env_spec.id
        elif hasattr(dataset, 'env_spec'):
            env_id = dataset.env_spec.id if hasattr(dataset.env_spec, 'id') else str(dataset.env_spec)
        else:
            raise ValueError("Could not determine env_id from dataset.env_spec")
        print(f"Resolved environment ID: {env_id}")
    else:
        raise ValueError("Must provide either --env or (--dataset, --minari_env, --names)")

    # Load hyperparameters if requested
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

    device = args.device
    print(f"Using device: {device}")

    # Prepare policy kwargs
    policy_kwargs = {
        "log_std_init": hparam.get("log_std_init", 0.0),
        "activation_fn": nn.ReLU,
        "optimizer_kwargs": {"betas": (0.999, 0.999)}
    }
    print("Learning rate for value function training:", args.learning_rate or hparam.get("learning_rate", 3e-4))
    # Instantiate PPOExpert with the Minari dataset so we can use its V-trace offline rollouts
    model = PPOExpert(
        MlpPolicyExpert,
        env,
        verbose=1,
        learning_rate=hparam.get("learning_rate", 1e-4),
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
        minari_dataset=dataset,
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

    # Ensure value network params require grad
    value_params = []
    if hasattr(model.policy, "mlp_extractor") and hasattr(model.policy.mlp_extractor, "value_net"):
        for p in model.policy.mlp_extractor.value_net.parameters():
            p.requires_grad = True
            value_params.append(p)
    if hasattr(model.policy, "value_net"):
        for p in model.policy.value_net.parameters():
            p.requires_grad = True
            value_params.append(p)

    if len(value_params) == 0:
        raise RuntimeError("No value parameters found to train")

    lr = args.learning_rate if args.learning_rate is not None else hparam.get("learning_rate", 3e-4)
    optimizer = th.optim.Adam(value_params, lr=lr)

    # Training loop: process transitions from dataset until requested timesteps processed
    processed = 0
    target = max(1, args.timesteps)
    batch_size = args.batch_size or model.batch_size

    print(f"Starting V-trace value-only tuning for {target} transitions (batch_size={batch_size})")
    n_epochs = 100
    model._reinit_critic_ortho()
    for epoch in range(n_epochs):
        # Fill the expert buffer using Minari transitions and V-trace GAE
        model.make_offline_rollouts(callback=None, expert_buffer=model.expert_buffer, n_rollout_steps=model.n_steps)

        # Iterate minibatches from the expert buffer
        for samples in model.expert_buffer.get(batch_size=batch_size):
            obs, actions, values, log_probs, advantages, returns, log_probs_expert, offline_val = samples
            # Convert observations and actions to tensors on device
            # Expert buffer already returns torch.Tensors; handle both cases.
            if isinstance(obs, th.Tensor):
                obs_tensor = obs.to(model.device)
            else:
                obs_tensor = obs_as_tensor(obs, model.device)
            # Actions may need dtype conversion for discrete spaces
            if isinstance(model.action_space, gym.spaces.Discrete) or isinstance(model.action_space, gym.spaces.MultiDiscrete):
                actions_tensor = actions.long().to(model.device)
            else:
                actions_tensor = actions.to(model.device)

            # Evaluate actions to obtain current value predictions
            # evaluate_actions typically returns (values, log_prob, entropy)
            eval_out = model.policy.evaluate_actions(obs_tensor, actions_tensor)
            if isinstance(eval_out, tuple) and len(eval_out) >= 1:
                values_pred = eval_out[0].flatten()
            else:
                _, values_pred, _ = model.policy.forward(obs_tensor)
                values_pred = values_pred.flatten()

            returns_tensor = returns.to(model.device).flatten()

            # Compute value loss (MSE)
            value_loss = F.mse_loss(values_pred, returns_tensor)

            optimizer.zero_grad()
            value_loss.backward()
            optimizer.step()

            processed += obs.shape[0]
            if processed % max(1, batch_size * 100) == 0:
                print(f"Processed {processed} transitions, value_loss={value_loss.item():.6f}")
                print(f"Mean returns: {returns_tensor.mean().item():.4f}, mean values_pred: {values_pred.mean().item():.4f}")

    # Save tuned policy
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = args.model_path.replace(".zip", "_value_vtrace_tuned.zip")
    model.policy.save(save_path)
    print(f"Saved value-tuned (V-trace) policy to {save_path}")
