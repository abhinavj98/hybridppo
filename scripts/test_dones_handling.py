import inspect
import numpy as np
import torch as th
from gymnasium import spaces
from hybridppo.ppo_expert import ExpertRolloutBuffer


def run_test():
    n_envs = 2
    buffer_size = 4
    gamma = 1.0
    gae_lambda = 1.0

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    buf = ExpertRolloutBuffer(
        buffer_size,
        obs_space,
        action_space,
        device="cpu",
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=n_envs,
        rho_bar=1.0,
        c_bar=1.0,
    )

    # Design rewards and episode_starts so env0 terminates at step 1, env1 never terminates
    rewards_env0 = [0.0, 10.0, 0.0, 0.0]
    rewards_env1 = [1.0, 1.0, 1.0, 1.0]
    episode_starts = [
        np.array([True, True]),    # step0
        np.array([False, False]),  # step1
        np.array([True, False]),   # step2 -> env0 new episode (terminal at previous step)
        np.array([False, False]),  # step3
    ]

    # Helper to call add with compatible signature
    sig = inspect.signature(ExpertRolloutBuffer.add)
    add_params = [p.name for p in sig.parameters.values()]

    for step in range(buffer_size):
        obs = np.zeros((n_envs, 1), dtype=np.float32) + float(step)
        actions = np.zeros((n_envs, 1), dtype=np.float32)
        rewards = np.array([rewards_env0[step], rewards_env1[step]], dtype=np.float32)
        ep_start = episode_starts[step]
        values = th.zeros((n_envs,), dtype=th.float32)
        log_prob = th.zeros((n_envs,), dtype=th.float32)
        log_prob_expert = th.zeros((n_envs,), dtype=th.float32)

        # Build args in order expected by the signature
        args = []
        for name in add_params:
            if name == 'self':
                continue
            if name in ('obs', 'observations'):
                args.append(obs)
            elif name in ('action', 'actions'):
                args.append(actions)
            elif name in ('reward', 'rewards'):
                args.append(rewards)
            elif name in ('episode_start', 'episode_starts'):
                args.append(ep_start)
            elif name in ('value', 'values', 'old_values'):
                args.append(values)
            elif name in ('log_prob', 'old_log_prob'):
                args.append(log_prob)
            elif name == 'log_prob_expert':
                args.append(log_prob_expert)
            else:
                # Fallback: try to supply None
                args.append(None)

        # Call add with constructed args
        try:
            buf.add(*args)
        except TypeError:
            # final fallback: call with the minimal stable signature
            buf.add(obs, actions, rewards, ep_start, values, log_prob, log_prob_expert)

    # last_values and dones for the final timestep
    last_values = np.zeros((n_envs,), dtype=np.float32)
    dones = np.array([False, False])

    buf.compute_returns_and_advantage(last_values=last_values, dones=dones)

    print("Returns (shape step x env):")
    print(buf.returns)
    print("Advantages (shape step x env):")
    print(buf.advantages)

    # Expected returns given gamma=1.0
    expected_env0 = np.array([10.0, 10.0, 0.0, 0.0], dtype=np.float32)
    expected_env1 = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)

    env0 = buf.returns[:, 0]
    env1 = buf.returns[:, 1]

    print("env0 returns:", env0)
    print("env1 returns:", env1)

    assert np.allclose(env0, expected_env0, atol=1e-4), f"env0 returns mismatch: {env0} vs {expected_env0}"
    assert np.allclose(env1, expected_env1, atol=1e-4), f"env1 returns mismatch: {env1} vs {expected_env1}"

    print("Dones handling test PASSED")


if __name__ == "__main__":
    run_test()
