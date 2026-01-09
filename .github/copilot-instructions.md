# Copilot Instructions for hybrid-ppo

Purpose: Help AI agents be productive quickly in this RL codebase that blends online PPO with offline Minari datasets (hybrid training + BC). Keep changes surgical, follow Stable-Baselines3 (SB3) APIs, and use the provided data pipeline utilities.

## Big Picture
- Core idea: Train PPO online while mixing offline transitions from Minari.
  - `PPOExpert` mixes online rollouts with offline minibatches using V-trace-style caps and OOD down-weighting.
  - `PPOBC` is a lighter hybrid: standard PPO online + BC loss from offline actions.
- Data comes from Minari; helpers stream transitions per-episode to mimic on-policy shapes.
- Policies extend SB3 actor-critic; `MlpPolicyExpert` adds `forward_expert(obs, action)` for evaluating given actions.

## Key Modules
- Offline pipeline: `hybridppo/minari_helpers.py`
  - `get_dataset(dataset, env, names)` loads/composes datasets; prefers local at `hybridppo/__init__.py:DATASET_PATH`, otherwise downloads via Minari.
  - `MinariTransitionDataset` exposes transition indices; `MultiEpisodeSequentialSampler` yields sequential steps across multiple episodes/envs; `collate_env_batch` reshapes to `[batch_size, n_envs, ...]` expected by buffers.
- Algorithms:
  - `hybridppo/ppo_expert.py` (`PPOExpert`):
    - Builds `ExpertRolloutBuffer` including `log_prob_expert` and computes advantages with V-trace-style `rho_bar` and `c_bar`.
    - Splits each update minibatch into online/offline via `mix_ratio` and applies OOD down-weighting using a Mahalanobis distance proxy from `log_prob` and `log_std` (clamped to min −1.6).
    - Excludes dataloader/iterators from saved params to keep checkpoints portable.
    - `train_bc(...)` provides a standalone BC loop on Minari transitions with optional value warm start.
  - `hybridppo/ppo_bc.py` (`PPOBC`): Uses online PPO loss + simple offline BC loss (no V-trace / expert log-prob).
- Policies: `hybridppo/policies.py`
  - `MlpPolicyExpert.MlpExtractor` lets you configure separate `pi` and `vf` MLPs and optional LayerNorm.
  - `forward_expert(obs, action)` returns `(action, value, log_prob)` with the provided `action` and current policy.

## Hyperparams & Conventions
- Central config: `hparam.yml` (e.g., `Walker2d-v4`, `Walker2d-v4-hybrid`, `Walker2d-v4-bc-large`). Keys map directly to SB3/PPO init fields.
- **Pipeline hparam mapping**: `train_hybrid_pipeline.py` auto-resolves env names:
  - Input: `--env walker2d`
  - BC hparam: `Walker2d-v4-bc-large` (capitalized + `-v4` + `-bc-large`)
  - Hybrid hparam: `Walker2d-v4-hybrid` (capitalized + `-v4` + `-hybrid`)
  - If custom names needed, manually call `train_bc_expert.py` and `test_ppo_expert.py` with `--hparam` flags.
- **Best BC checkpoint selection**: Pipeline parses `bc_checkpoints/<dataset>/<env>/*_metrics.csv` (columns: `epoch,train_loss,val_loss`), finds epoch with lowest `val_loss`, and passes that `.zip` to hybrid PPO as `--bc_policy`.
- Devices: many scripts support `--device auto` (mps→cuda→cpu). Some value-finetune and dataloading paths ensure `float32` for MPS.
- Actions: when `spaces.Discrete`/`MultiDiscrete`, cast actions to `.long()` before loss (see `ppo_expert.py`, `ppo_bc.py`).
- Dataloader output shapes are `[n_steps, n_envs, ...]`; training code flattens as needed.

## Typical Workflows

### 1. Automated BC → Hybrid PPO Pipeline (Recommended)
Trains BC (if needed), finds lowest val_loss epoch, warmstarts hybrid PPO, logs pipeline run.
```bash
python training_files/train_hybrid_pipeline.py \
  --dataset mujoco \
  --env walker2d \
  --names expert-v0 \
  --seed 42 \
  --num_bc_epochs 50 \
  --device auto
```
Options:
- `--skip_bc`: Reuse existing best BC checkpoint
- `--force_bc_retrain`: Force retrain BC
- `--mix_ratio 0.5`, `--rho_bar 1.0`, `--c_bar 0.95`: V-trace and mixing params
- Outputs: BC checkpoint, hybrid PPO checkpoint, pipeline summary → `logs/pipeline_runs.json`

### 2. Online PPO only
```bash
python training_files/test_ppo.py --dataset mujoco --env walker2d --name expert-v0 --hparam Walker2d-v4 [--wandb]
```

### 3. Hybrid PPO (offline+online, V-trace + OOD weighting)
```bash
python training_files/test_ppo_expert.py \
  --dataset mujoco --env walker2d --names expert-v0 expert-v1 \
  --hparam Walker2d-v4-hybrid --mix_ratio 0.5 --rho_bar 1.0 --c_bar 0.95 \
  --log_std_subtract 0.0 --save_file runA
```
With optional BC warm start:
```bash
python training_files/test_ppo_expert.py \
  --dataset mujoco --env walker2d --names expert-v0 \
  --hparam Walker2d-v4-hybrid --bc_policy bc_checkpoints/mujoco/walker2d/bc_walker2d_..._epoch50.zip \
  --save_file runA_with_bc
```

### 4. Behavioral Cloning only (fast pretraining + val split)
```bash
python training_files/train_bc_expert.py \
  --dataset mujoco --env walker2d --names expert-v0 \
  --hparam Walker2d-v4-bc-large --bc_epochs 50 --bc_batch_size 128 \
  --bc_coeff 0.005 --warm_start_steps 100 --save_dir bc_checkpoints
```

## Datasets
- Local path root is fixed at `hybridppo/__init__.py:DATASET_PATH`.
- `get_dataset` resolves each `name` under `{DATASET_PATH}/{dataset}/{env}/{name}`; if missing, downloads (`minari.download_dataset`) and can combine multiple names into a single dataset.
- Environments are recovered from datasets via `dataset.env_spec` (Gymnasium id). Vector envs built using `stable_baselines3.common.env_util.make_vec_env`.

## Logging & Checkpoints
- Weights & Biases: `test_ppo.py` supports `--wandb`; hybrid scripts initialize W&B runs with merged `hparam.yml`.
- TensorBoard logs: stored under `tb_test/online` or `tb_test/hybrid/...` depending on script.
- Checkpoints:
  - PPO scripts use `model.save(<save_file>_i)` (zip). `PPOExpert` excludes dataloaders via `_excluded_save_params`.
  - BC training saves to `bc_checkpoints/<dataset>/<env>/<name>_epochN with avg loss X.XXXX.zip` and writes metrics CSV alongside.
  - `eval_model.py` loads with `--algo {PPO|PPOBC|PPOExpert}` and `--model_path path/without/.zip` and evaluates for `--episodes`.

## Implementation Tips
- Respect SB3 APIs: use provided policy and algorithm classes; don’t mutate buffer internals except where already extended (e.g., `ExpertRolloutBuffer`).
- Keep dataloader contracts: `MultiEpisodeSequentialSampler` + `collate_env_batch` must produce aligned per-env trajectories over `n_steps`.
- Numerical stability: clamp log-ratio terms and `log_std`; normalize advantages only where existing code does.
- When extending losses, mirror the online/offline split used in `PPOExpert.train_ppo_offline` and preserve logging keys.

## Quick Examples
- Evaluate a saved PPOExpert checkpoint:
  - `python eval_model.py --dataset mujoco --env walker2d --name expert-v0 --algo PPOExpert --model_path logs/best_model`
- Sanity-check dataloader throughput:
  - `python minari_transition_dataset_tests.py --dataset mujoco --env walker2d --name expert-v0`

If anything above is unclear (e.g., dataset roots, expected batch shapes, or logging paths), please comment which section needs more detail and the scenario you’re targeting (BC, PPO, or hybrid).

The final aim of this repo is to make an automated benchmarking of the hybrid offline+online RL over different minari tasks with a set of seeds. Eventually this would also include hyperparameter sweeps and additional algorithms.

Do not use emojis and keep the code professional.