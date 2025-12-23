import unittest
import torch
from hybridppo.minari_helpers import MinariTransitionDataset, MultiEpisodeSequentialSampler, collate_env_batch

class DummyEpisode:
    def __init__(self, T):
        self.observations = [torch.ones(3)*i for i in range(T+1)]
        self.actions = [torch.zeros(2) for _ in range(T)]
        self.rewards = [1.0]*T
        self.terminations = [False]*(T-1) + [True]
        self.truncations = [False]*T

class DummyMinariDataset:
    def __init__(self, n_eps, T):
        self.episodes = [DummyEpisode(T) for _ in range(n_eps)]

    def __getitem__(self, idx):
        return self.episodes[idx]

    def __len__(self):
        return len(self.episodes)

class TestMinariTransition(unittest.TestCase):
    def test_dataset_indexing(self):
        dataset = DummyMinariDataset(5, 10)
        trans_dataset = MinariTransitionDataset(dataset)
        self.assertEqual(len(trans_dataset), 5 * 9)  # (T - 1) per episode
        sample = trans_dataset[0]
        self.assertIn("observations", sample)
        self.assertEqual(sample["observations"].shape, torch.Size([3]))

    def test_sampler_batch_shape(self):
        dataset = DummyMinariDataset(10, 10)
        trans_dataset = MinariTransitionDataset(dataset)
        sampler = MultiEpisodeSequentialSampler(trans_dataset, n_envs=4, batch_size=8, seed=42)
        iterator = iter(sampler)
        batch_indices = next(iterator)
        self.assertEqual(len(batch_indices), 32)  # 4 envs * 8 batch

    def test_collate(self):
        dataset = DummyMinariDataset(10, 10)
        trans_dataset = MinariTransitionDataset(dataset)
        sampler = MultiEpisodeSequentialSampler(trans_dataset, n_envs=2, batch_size=4, seed=0)
        indices = next(iter(sampler))
        batch = [trans_dataset[idx] for idx in indices]
        out = collate_env_batch(batch, 2, 4)
        self.assertEqual(out["observations"].shape, torch.Size([4, 2, 3]))

    from hybridppo.minari_helpers import MinariTransitionDataset, MultiEpisodeSequentialSampler, collate_env_batch
    from torch.utils.data import DataLoader

class TestMinariOfflineRL(unittest.TestCase):

    def test_transition_dataset_indexing(self):
        dummy_data = DummyMinariDataset()
        dataset = MinariTransitionDataset(dummy_data)
        expected_len = 4 * (20)  # 4 episodes × (T-1) steps
        self.assertEqual(len(dataset), expected_len)

        sample = dataset[0]
        self.assertIn("observations", sample)
        self.assertEqual(sample["observations"].shape, torch.Size([3]))

    def test_sampler_ordering_and_collation(self):
        dummy_data = DummyMinariDataset()
        dataset = MinariTransitionDataset(dummy_data)
        sampler = MultiEpisodeSequentialSampler(dataset, n_envs=2, batch_size=5, seed=42)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: collate_env_batch(b, n_envs=2, batch_size=5),
            num_workers=0,
        )
        batch = next(iter(dataloader))

        # Expect shape: [batch_size, n_envs, ...]
        self.assertEqual(batch["observations"].shape, (5, 2, 3))

        # Ensure temporal order is preserved in each env
        for env_id in range(2):
            step_ids = batch["step_ids"][:, env_id].tolist()
            self.assertEqual(step_ids, list(range(5)), f"Step IDs not in order: {step_ids}")

            episode_ids = batch["episode_ids"][:, env_id].tolist()
            self.assertTrue(all(ep == episode_ids[0] for ep in episode_ids), "Mixed episode IDs in env")

    def test_dataloader_wraparound(self):
        dummy_data = DummyMinariDataset()
        dataset = MinariTransitionDataset(dummy_data)
        sampler = MultiEpisodeSequentialSampler(dataset, n_envs=2, batch_size=20, seed=1)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: collate_env_batch(b, n_envs=2, batch_size=20),
            num_workers=0,
        )
        iterator = iter(dataloader)
        for _ in range(3):  # Run multiple times to ensure sampler wraparound
            batch = next(iterator)
            self.assertEqual(batch["observations"].shape[0], 20)

# class DummyEpisode:
#     def __init__(self, T, id):
#         self.observations = [torch.ones(3) * (id * 1000 + i) for i in range(T + 1)]
#         self.actions = [torch.ones(2) * i for i in range(T)]
#         self.rewards = [1.0] * T
#         self.terminations = [False] * (T - 1) + [True]
#         self.truncations = [False] * T
#
# class DummyMinariDataset:
#     def __init__(self, n_eps=4, T=100):
#         self.episodes = [DummyEpisode(T, id=i) for i in range(n_eps)]
#
#     def __getitem__(self, idx):
#         return self.episodes[idx]
#
#     def __len__(self):
#         return len(self.episodes)

if __name__ == "__main__":
    unittest.main()

    import torch
    from torch.utils.data import DataLoader

    # ========== Dummy Minari-Like Dataset ==========
    class DummyEpisode:
        def __init__(self, T, id):
            self.observations = [torch.ones(3) * (id * 1000 + i) for i in range(T + 1)]
            self.actions = [torch.ones(2) * i for i in range(T)]
            self.rewards = [1.0] * T
            self.terminations = [False] * (T - 1) + [True]
            self.truncations = [False] * T

    class DummyMinariDataset:
        def __init__(self, n_eps=4, T=100):
            self.episodes = [DummyEpisode(T, id=i) for i in range(n_eps)]

        def __getitem__(self, idx):
            return self.episodes[idx]

        def __len__(self):
            return len(self.episodes)



    # ====== Use Your Implementations ======
    # Assume MinariTransitionDataset, MultiEpisodeSequentialSampler, collate_env_batch are defined as above

    # ---- Setup ----
    n_envs = 4
    batch_size = 8
    dummy_dataset = DummyMinariDataset(n_eps=4, T=100)
    transition_dataset = MinariTransitionDataset(dummy_dataset)
    sampler = MultiEpisodeSequentialSampler(transition_dataset, n_envs=n_envs, batch_size=batch_size, seed=123)

    # ---- Load 1 batch ----
    loader = DataLoader(
        transition_dataset,
        batch_sampler=sampler,
        collate_fn=lambda b: collate_env_batch(b, n_envs=n_envs, batch_size=batch_size),
        num_workers=0
    )

    # ---- Fetch and print ----
    batch = next(iter(loader))

    # Print episode and step IDs to verify temporal structure
    print("== Episode IDs ==")
    print(batch["episode_ids"])

    print("== Step IDs ==")
    print(batch["step_ids"])

    # Optional: check if transitions in each env are temporally increasing
    for env_id in range(n_envs):
        steps = batch["step_ids"][:, env_id].tolist()
        print(f"Env {env_id}: Step IDs: {steps}")
        assert steps == sorted(steps), f"Env {env_id} steps not ordered!"
