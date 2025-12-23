import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from argparse import ArgumentParser
from hybridppo.minari_helpers import get_dataset, get_eval_environment

# Import all possible algorithms
from stable_baselines3 import PPO
from hybridppo.ppo_bc import PPOBC
from hybridppo.ppo_expert import PPOExpert  # assuming you have this module
from hybridppo.policies import MlpPolicyExpert


ALGO_MAP = {
    "PPO": PPO,
    "PPOBC": PPOBC,
    "PPOExpert": PPOExpert
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D4RL")
    parser.add_argument("--env", type=str, default="door")
    parser.add_argument("--name", type=str, default="human")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file (without .zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--algo", type=str, required=True, choices=["PPO", "PPOBC", "PPOExpert"], help="RL algorithm class")

    args = parser.parse_args()

    dataset_name = f"{args.dataset}/{args.env}/{args.name}"

    # Load dataset to get environment spec
    dataset = get_dataset(args.dataset, args.env, args.name)
    eval_env = get_eval_environment(dataset, render_mode="human")

    # Pick the algorithm
    AlgoClass = ALGO_MAP[args.algo]

    # Load model
    model = AlgoClass.load(args.model_path, env=eval_env)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.episodes, render=False)
    print(f"✅ Evaluated model '{args.model_path}' using {args.algo}")
    print(f"📈 Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
