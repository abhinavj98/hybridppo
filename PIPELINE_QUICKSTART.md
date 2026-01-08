Quick Start: Hybrid PPO Pipeline
================================

The pipeline automates BC training -> best checkpoint selection -> hybrid PPO training.

BASIC USAGE
-----------

Minimal command:
  python training_files/train_hybrid_pipeline.py \
    --dataset mujoco \
    --env walker2d \
    --names expert-v0 \
    --hparam_bc Walker2d-v4-bc-large \
    --hparam_hybrid Walker2d-v4-hybrid \
    --seed 42

Full command (with custom V-trace and BC settings):
  python training_files/train_hybrid_pipeline.py \
    --dataset mujoco \
    --env walker2d \
    --names expert-v0 expert-v1 \
    --hparam_bc Walker2d-v4-bc-large \
    --hparam_hybrid Walker2d-v4-hybrid \
    --seed 42 \
    --num_bc_epochs 50 \
    --bc_batch_size 128 \
    --bc_coeff 0.005 \
    --warm_start_steps 100 \
    --mix_ratio 0.5 \
    --rho_bar 1.0 \
    --c_bar 0.95 \
    --log_std_subtract 0.0 \
    --device auto

Reuse existing BC checkpoint:
  python training_files/train_hybrid_pipeline.py \
    --dataset mujoco \
    --env walker2d \
    --names expert-v0 \
    --hparam_bc Walker2d-v4-bc-large \
    --hparam_hybrid Walker2d-v4-hybrid \
    --seed 42 \
    --skip_bc

Force retrain BC (even if checkpoint exists):
  python training_files/train_hybrid_pipeline.py \
    --dataset mujoco \
    --env walker2d \
    --names expert-v0 \
    --hparam_bc Walker2d-v4-bc-large \
    --hparam_hybrid Walker2d-v4-hybrid \
    --seed 42 \
    --force_bc_retrain


ARGUMENTS
---------

Required:
  --dataset STR           Dataset type (e.g., mujoco)
  --env STR               Environment name (e.g., walker2d, humanoid, ant)
  --names STR+            Dataset variant names (e.g., expert-v0 expert-v1)
  --hparam_bc STR         Hparam key for BC training (e.g., Walker2d-v4-bc-large)
  --hparam_hybrid STR     Hparam key for hybrid PPO (e.g., Walker2d-v4-hybrid)

Optional:
  --seed INT              Random seed (default: 42)
  --skip_bc               Skip BC training if checkpoint exists
  --force_bc_retrain      Force BC retraining
  --num_bc_epochs INT     BC epochs (default: 50)
  --bc_batch_size INT     BC batch size (default: 128)
  --bc_coeff FLOAT        BC loss coefficient (default: 0.005)
  --warm_start_steps INT  Value finetune steps (default: 100)
  --log_interval INT      BC logging interval (default: 10)
  --save_every_epochs INT Save BC checkpoint every N epochs (default: 5)
  --mix_ratio FLOAT       Offline/online split, 0..1 (default: 0.5)
  --rho_bar FLOAT         V-trace rho_bar cap (default: 1.0)
  --c_bar FLOAT           V-trace c_bar cap (default: 0.95)
  --log_std_subtract FLOAT Subtract from log_std (default: 0.0)
  --device STR            Device: auto|cpu|cuda|mps (default: auto)


HPARAM KEYS
-----------

Available hparam keys in hparam.yml:

  Walker2d:
    BC:     Walker2d-v4-bc-large
    Hybrid: Walker2d-v4-hybrid

  Humanoid:

  humanoid  ->  Humanoid-v4-bc-large  (BC)
                Humanoid-v4-hybrid      (Hybrid PPO)

  ant       ->  Ant-v4-bc-large       (BC)
                Ant-v4-hybrid           (Hybrid PPO)

These keys must exist in hparam.yml for validation to pass.


WORKFLOW
--------

1. Checks for existing BC checkpoint
   - If found, asks whether to retrain or reuse
   - Use --skip_bc to always reuse
   - Use --force_bc_retrain to always retrain

2. Trains BC (if needed)
   - Saves checkpoints every N epochs
   - Writes metrics CSV: epoch, train_loss, val_loss

3. Finds best BC checkpoint
   - Parses *_metrics.csv files in bc_checkpoints/<dataset>/<env>/
   - Selects checkpoint with lowest val_loss

4. Trains hybrid PPO with BC warmstart
   - Initializes policy from best BC checkpoint
   - Uses V-trace caps and OOD down-weighting
   - Logs to TensorBoard and W&B (if enabled)

5. Logs pipeline run
   - Saves metadata to logs/pipeline_runs.json
   - Includes BC loss, epochs, hybrid save file, hparams


OUTPUTS
-------

BC Training:
  bc_checkpoints/<dataset>/<env>/<name>_epoch{N} with avg loss X.XXXX.zip
  bc_checkpoints/<dataset>/<env>/<name>_metrics.csv (columns: epoch,train_loss,val_loss)

Hybrid PPO:
  hybrid_ppo_<env>_<timestamp>.zip (or custom via --save_file)
  tb_test/hybrid/<dataset>/<env>/<save_file>/

Pipeline Log:
  logs/pipeline_runs.json (append-only JSON with run metadata)


TESTING
-------

Run unit tests:
  cd training_files
  python test_train_hybrid_pipeline.py

Run integration test:
  python test_pipeline_integration.py

All tests should pass before using pipeline in production.


TROUBLESHOOTING
---------------

"ERROR: BC hparam key 'Walker2d-v4-bc-large' not found in hparam.yml"
  -> Add the missing key to hparam.yml (see existing entries for format)

"Pipeline failed: No BC checkpoint available"
  -> BC training failed; check logs for errors, retry with --force_bc_retrain

"No metrics CSV files found in bc_checkpoints/mujoco/walker2d"
  -> BC training didn't save checkpoints; check save_every_epochs and num_bc_epochs

"BC checkpoint dir not found: bc_checkpoints/mujoco/walker2d"
  -> This is normal for first run; pipeline will train BC from scratch
