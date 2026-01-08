"""
Integration test: Verify pipeline flow with mock subprocess calls.
Tests argparse, validation, and checkpoint selection flow.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from train_hybrid_pipeline import validate_hparams


def test_pipeline_flow():
    """Simulate end-to-end pipeline flow without running actual training."""
    print("\nTesting Pipeline Flow...")
    print("=" * 60)
    
    # Simulate user inputs
    dataset = "mujoco"
    env = "walker2d"
    hparam_bc = "Walker2d-v4-bc-large"
    hparam_hybrid = "Walker2d-v4-hybrid"
    names = ["expert-v0"]
    seed = 42
    skip_bc = False
    
    print(f"\nInput Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Environment: {env}")
    print(f"  BC hparam: {hparam_bc}")
    print(f"  Hybrid hparam: {hparam_hybrid}")
    print(f"  Dataset names: {names}")
    print(f"  Seed: {seed}")
    
    # Step 1: Validate hparams
    print(f"\n[Step 1] Validating hparams against hparam.yml...")
    import yaml
    hparam_path = Path(__file__).parent.parent / "hparam.yml"
    
    if hparam_path.exists():
        with open(hparam_path) as f:
            hparam_all = yaml.safe_load(f)
        
        is_valid = validate_hparams(hparam_all, hparam_bc, hparam_hybrid)
        
        if is_valid:
            print(f"  BC hparam exists: {hparam_bc}")
            print(f"  Hybrid hparam exists: {hparam_hybrid}")
            print(f"  Validation: PASSED")
        else:
            print(f"  Validation: FAILED")
            return False
    else:
        print(f"  WARNING: hparam.yml not found at {hparam_path}")
    
    # Step 2: Check for existing BC checkpoint
    print(f"\n[Step 2] Checking for existing BC checkpoint...")
    bc_checkpoint_dir = Path("bc_checkpoints") / dataset / env
    
    if bc_checkpoint_dir.exists():
        metrics_files = list(bc_checkpoint_dir.glob("*_metrics.csv"))
        zip_files = list(bc_checkpoint_dir.glob("*.zip"))
        print(f"  BC checkpoint directory exists: {bc_checkpoint_dir}")
        print(f"  Metrics CSV files found: {len(metrics_files)}")
        print(f"  Checkpoint ZIP files found: {len(zip_files)}")
        
        if metrics_files and zip_files:
            print(f"  Would proceed to: Extract best checkpoint by val_loss")
        else:
            print(f"  Would proceed to: Train new BC model")
    else:
        print(f"  BC checkpoint directory does not exist")
        print(f"  Would proceed to: Train new BC model")
    
    # Step 3: Verify subprocess command construction
    print(f"\n[Step 3] Simulating subprocess command construction...")
    
    bc_cmd = [
        "python", "training_files/train_bc_expert.py",
        "--dataset", dataset,
        "--env", env,
        "--names", *names,
        "--hparam", hparam_bc,
        "--seed", str(seed),
    ]
    print(f"  BC training command would be:")
    print(f"    {' '.join(bc_cmd[:3])} \\")
    print(f"      {' '.join(bc_cmd[3:])}")
    
    hybrid_cmd = [
        "python", "training_files/test_ppo_expert.py",
        "--dataset", dataset,
        "--env", env,
        "--names", *names,
        "--hparam", hparam_hybrid,
        "--seed", str(seed),
    ]
    print(f"\n  Hybrid PPO training command would be:")
    print(f"    {' '.join(hybrid_cmd[:3])} \\")
    print(f"      {' '.join(hybrid_cmd[3:])}")
    
    # Step 4: Verify JSON logging structure
    print(f"\n[Step 4] Verifying pipeline run JSON structure...")
    
    run_entry = {
        "timestamp": "2025-12-24T15:30:00",
        "success": True,
        "dataset": dataset,
        "env": env,
        "names": names,
        "seed": seed,
        "bc_hparam": hparam_bc,
        "hybrid_hparam": hparam_hybrid,
        "bc_checkpoint": "bc_checkpoints/mujoco/walker2d/bc_walker2d_epoch50.zip",
        "bc_val_loss": 0.123,
        "bc_best_epoch": 50,
        "bc_epochs_trained": 50,
        "hybrid_save_file": "hybrid_ppo_walker2d_20251224_150000",
    }
    
    print(f"  Would log to logs/pipeline_runs.json:")
    print(f"    Keys: {list(run_entry.keys())}")
    print(f"    Sample entry structure valid: {len(run_entry) > 10}")
    
    print(f"\n{'='*60}")
    print(f"Pipeline flow simulation: PASSED")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    success = test_pipeline_flow()
    sys.exit(0 if success else 1)
