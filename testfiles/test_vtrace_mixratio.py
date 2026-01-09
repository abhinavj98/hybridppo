"""
Lightweight tests for V-trace caps in ExpertRolloutBuffer and mix_ratio splitting logic.
Run directly with: python training_files/test_vtrace_mixratio.py
"""
import os
import sys
import numpy as np
import torch as th
from gymnasium import spaces

# Ensure repository root is on sys.path to import `hybridppo`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybridppo.ppo_expert import ExpertRolloutBuffer


def build_dummy_expert_buffer(buffer_size=4, obs_dim=2, act_dim=1, gamma=0.99, gae_lambda=0.95,
                              rho_bar=1.0, c_bar=0.95):
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    buf = ExpertRolloutBuffer(
        buffer_size,
        obs_space,
        act_space,
        device="cpu",
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=1,
        rho_bar=rho_bar,
        c_bar=c_bar,
    )
    buf.reset()
    # Deterministic synthetic trajectory
    # Observations/actions
    for step in range(buffer_size):
        obs = np.zeros((1, obs_dim), dtype=np.float32) + step
        act = np.ones((1, act_dim), dtype=np.float32) * 0.5
        reward = np.array([1.0], dtype=np.float32)  # constant reward
        episode_start = np.array([False], dtype=bool) if step > 0 else np.array([True], dtype=bool)
        value = th.zeros((1, 1), dtype=th.float32)
        log_prob_current = th.full((1, 1), 10.0, dtype=th.float32)  # very high
        log_prob_expert = th.full((1, 1), -10.0, dtype=th.float32)  # very low -> huge ratio
        buf.add(obs, act, reward, episode_start, value, log_prob_current, log_prob_expert)
    # Last values and dones
    last_values = np.array([0.0], dtype=np.float32)
    dones = np.array([False], dtype=bool)
    buf.compute_returns_and_advantage(last_values=last_values, dones=dones)
    return buf


def test_vtrace_caps_reduce_advantages():
    # Uncapped (large caps)
    uncapped = build_dummy_expert_buffer(rho_bar=100.0, c_bar=100.0)
    # Strongly capped
    capped = build_dummy_expert_buffer(rho_bar=0.5, c_bar=0.4)
    # Compare mean advantages magnitude
    adv_uncapped = np.abs(uncapped.advantages).mean()
    adv_capped = np.abs(capped.advantages).mean()
    assert adv_capped < adv_uncapped, (
        f"Expected capped advantages < uncapped: {adv_capped} !< {adv_uncapped}")
    print("test_vtrace_caps_reduce_advantages: PASS")


def compute_split(batch_size: int, mix_ratio: float):
    mix_ratio = max(0.0, min(1.0, mix_ratio))
    print(f"Computing split for batch_size={batch_size}, mix_ratio={mix_ratio}")
    offline_bs = max(1, int(batch_size * mix_ratio))
    online_bs = max(1, batch_size - offline_bs)
    return offline_bs, online_bs


def test_mix_ratio_split_logic():
    # A few scenarios
    for batch_size in [8, 32, 33, 64]:
        prev_offline = None
        for mix_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            offline_bs, online_bs = compute_split(batch_size, mix_ratio)
            # Both splits must be at least 1 and at most batch_size
            assert 1 <= offline_bs <= batch_size
            assert 1 <= online_bs <= batch_size
            # Offline size should be monotonic non-decreasing with mix_ratio
            if prev_offline is not None:
                assert offline_bs >= prev_offline, (
                    f"offline_bs not monotonic: {offline_bs} < {prev_offline} for mix_ratio={mix_ratio}")
            prev_offline = offline_bs
    print("test_mix_ratio_split_logic: PASS")


if __name__ == "__main__":
    test_vtrace_caps_reduce_advantages()
    test_mix_ratio_split_logic()
    print("All tests passed.")
