"""
Automated BC -> Hybrid PPO training pipeline.

Workflow:
1. Accept explicit hparam keys via --hparam_bc and --hparam_hybrid
2. Check for existing BC checkpoint; reuse if --skip_bc
3. Train BC if needed (with --seed)
4. Find best BC checkpoint by parsing val_loss from _metrics.csv
5. Train hybrid PPO with BC warmstart (--bc_policy)
6. Log pipeline run to logs/pipeline_runs.json
"""

import os
import sys
import json
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd





def validate_hparams(hparam_all: dict, bc_key: str, hybrid_key: str) -> bool:
    """Check that both BC and hybrid hparam keys exist."""
    if bc_key not in hparam_all:
        print(f"ERROR: BC hparam key '{bc_key}' not found in hparam.yml")
        return False
    if hybrid_key not in hparam_all:
        print(f"ERROR: Hybrid hparam key '{hybrid_key}' not found in hparam.yml")
        return False
    return True


def find_best_bc_checkpoint(dataset: str, env: str, names: list) -> Optional[Tuple[Path, float, int]]:
    """
    Find BC checkpoint with lowest val_loss by parsing _metrics.csv files.
    
    Returns:
        (checkpoint_path, best_val_loss, best_epoch) or None if no checkpoints found
    """
    bc_dir = Path("./bc_checkpoints") / dataset / env / ''.join(names)
    if not bc_dir.exists():
        print(f"BC checkpoint dir not found: {bc_dir}")
        return None
    
    metrics_files = list(bc_dir.glob("*_metrics.csv"))
    if not metrics_files:
        print(f"No metrics CSV files found in {bc_dir}")
        return None
    
    best_loss = float('inf')
    best_ckpt = None
    best_epoch = None
    
    for metrics_csv in metrics_files:
        try:
            df = pd.read_csv(metrics_csv)
            if "val_loss" not in df.columns:
                print(f"WARNING: val_loss column not found in {metrics_csv.name}")
                continue
            
            # Find row with minimum val_loss
            min_idx = df["val_loss"].idxmin()
            min_loss = df.loc[min_idx, "val_loss"]
            epoch = int(df.loc[min_idx, "epoch"])
            print(f"Parsed {metrics_csv.name}: epoch {epoch}, val_loss = {min_loss:.6f}")
            if min_loss < best_loss:
                best_loss = min_loss
                best_epoch = epoch
                
                # Find corresponding .zip checkpoint
                # Pattern: bc_*.zip files; look for epoch{N} in name or order by mtime
                zip_files = sorted(bc_dir.glob("*.zip"))
                print(f"Found {len(zip_files)} .zip files for checkpoint matching")
                if zip_files:

                    # Match epoch number in filename if present, else use most recent
                    for zf in zip_files:
                        if f"epoch{epoch}" in zf.name:
                            best_ckpt = zf
                            break
                    if not best_ckpt:
                        # Fallback: use most recent .zip (could be the best one)
                        best_ckpt = max(zip_files, key=lambda p: p.stat().st_mtime)
        
        except Exception as e:
            print(f"ERROR parsing {metrics_csv.name}: {e}")
            continue
    
    if best_ckpt:
        print(f"Found best BC checkpoint: {best_ckpt.name}")
        print(f"   Epoch {best_epoch}, val_loss = {best_loss:.6f}")
        return (best_ckpt, best_loss, best_epoch)
    
    return None


def run_bc_training(args, hparam_bc_key: str) -> Optional[Path]:
    """
    Run train_bc_expert.py subprocess. Returns best checkpoint path if successful.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: Training Behavioral Cloning")
    print(f"{'='*60}")
    
    cmd = [
        "python",
        "training_files/train_bc_expert.py",
        "--dataset", args.dataset,
        "--env", args.env,
        "--names", *args.names,
        "--hparam", hparam_bc_key,
        "--bc_epochs", str(args.num_bc_epochs),
        "--bc_batch_size", str(args.bc_batch_size),
        "--bc_coeff", str(args.bc_coeff),
        "--warm_start_steps", str(args.warm_start_steps),
        "--save_dir", "bc_checkpoints",
        "--log_interval", str(args.log_interval),
        "--save_every_epochs", str(args.save_every_epochs),
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"BC training completed successfully")
        
        # Find best checkpoint from newly trained run
        best = find_best_bc_checkpoint(args.dataset, args.env)
        if best:
            return best[0]
        else:
            print("ERROR: BC training completed but no checkpoint found!")
            return None
    
    except subprocess.CalledProcessError as e:
        print(f"BC training failed with exit code {e.returncode}")
        return None


def run_hybrid_ppo_training(
    args,
    hparam_hybrid_key: str,
    bc_checkpoint: Path,
    best_val_loss: float
) -> Optional[str]:
    """
    Run test_ppo_expert.py with BC warmstart. Returns save_file name if successful.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2: Training Hybrid PPO with BC Warmstart")
    print(f"{'='*60}")
    print(f"BC checkpoint: {bc_checkpoint.name}")
    print(f"BC val_loss: {best_val_loss:.6f}")
    print(f"Hybrid hparam: {hparam_hybrid_key}")
    
    # Generate unique save file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = f"hybrid_ppo_{args.env}_{timestamp}"
    
    cmd = [
        "python",
        "training_files/train_ppo_expert.py",
        "--dataset", args.dataset,
        "--env", args.env,
        "--names", *args.names,
        "--hparam", hparam_hybrid_key,
        "--bc_policy", str(bc_checkpoint),
        "--save_file", save_file,
        "--mix_ratio", str(args.mix_ratio),
        "--rho_bar", str(args.rho_bar),
        "--c_bar", str(args.c_bar),
        "--log_std_subtract", str(args.log_std_subtract),
        "--seed", str(args.seed),
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"Hybrid PPO training completed successfully")
        return save_file
    
    except subprocess.CalledProcessError as e:
        print(f"Hybrid PPO training failed with exit code {e.returncode}")
        return None


def log_pipeline_run(
    args,
    hparam_keys: Dict[str, str],
    bc_checkpoint: Path,
    bc_val_loss: float,
    bc_epoch: int,
    hybrid_save_file: Optional[str],
    success: bool
):
    """
    Log pipeline run metadata to logs/pipeline_runs.json
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "pipeline_runs.json"
    runs = []
    
    if log_file.exists():
        try:
            with open(log_file) as f:
                runs = json.load(f)
        except:
            runs = []
    
    run_entry = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "dataset": args.dataset,
        "env": args.env,
        "names": args.names,
        "seed": args.seed,
        "bc_hparam": hparam_keys["bc"],
        "hybrid_hparam": hparam_keys["hybrid"],
        "bc_checkpoint": str(bc_checkpoint),
        "bc_val_loss": float(bc_val_loss),
        "bc_best_epoch": int(bc_epoch),
        "bc_epochs_trained": args.num_bc_epochs,
        "hybrid_save_file": hybrid_save_file,
        "tensorboard_log": f"./tb_test/hybrid/{args.dataset}/{args.env}/{hybrid_save_file}" if hybrid_save_file else None,
        "mix_ratio": args.mix_ratio,
        "rho_bar": args.rho_bar,
        "c_bar": args.c_bar,
        "log_std_subtract": args.log_std_subtract,
    }
    
    runs.append(run_entry)
    
    with open(log_file, 'w') as f:
        json.dump(runs, f, indent=2)
    
    print(f"Pipeline run logged to {log_file}")


def main():
    parser = ArgumentParser(
        description="Automated BC warmstart -> Hybrid PPO training pipeline"
    )
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., mujoco)")
    parser.add_argument("--env", type=str, required=True,
                        help="Environment name (e.g., walker2d, humanoid)")
    parser.add_argument("--names", nargs='+', required=True,
                        help="Dataset variant names (e.g., expert-v0 expert-v1)")
    parser.add_argument("--hparam_bc", type=str, required=True,
                        help="Hparam key for BC training (e.g., Walker2d-v4-bc-large)")
    parser.add_argument("--hparam_hybrid", type=str, required=True,
                        help="Hparam key for hybrid PPO training (e.g., Walker2d-v4-hybrid)")
    
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for BC and hybrid training")
    
    # BC training options
    parser.add_argument("--skip_bc", action="store_true",
                        help="Skip BC training; reuse existing best checkpoint")
    parser.add_argument("--force_bc_retrain", action="store_true",
                        help="Force retrain BC even if checkpoint exists")
    parser.add_argument("--num_bc_epochs", type=int, default=50,
                        help="Number of BC training epochs")
    parser.add_argument("--bc_batch_size", type=int, default=128,
                        help="BC batch size")
    parser.add_argument("--bc_coeff", type=float, default=0.005,
                        help="BC loss coefficient")
    parser.add_argument("--warm_start_steps", type=int, default=100,
                        help="Value finetune steps in BC training")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="BC logging interval")
    parser.add_argument("--save_every_epochs", type=int, default=5,
                        help="Save BC checkpoint every N epochs")
    
    # Hybrid PPO options
    parser.add_argument("--mix_ratio", type=float, default=0.5,
                        help="Offline/online minibatch split (0..1)")
    parser.add_argument("--rho_bar", type=float, default=1.0,
                        help="V-trace rho_bar cap")
    parser.add_argument("--c_bar", type=float, default=0.95,
                        help="V-trace c_bar cap")
    parser.add_argument("--log_std_subtract", type=float, default=0.0,
                        help="Subtract from log_std after each update")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for training")
    
    args = parser.parse_args()
    
    # Load hparam.yml
    hparam_path = Path(__file__).parent.parent / "hparam.yml"
    with open(hparam_path) as f:
        hparam_all = yaml.safe_load(f)
    
    # Use provided hparam keys directly
    hparam_keys = {"bc": args.hparam_bc, "hybrid": args.hparam_hybrid}
    print(f"\nEnvironment: {args.env}")
    print(f"   BC hparam key: {hparam_keys['bc']}")
    print(f"   Hybrid hparam key: {hparam_keys['hybrid']}")
    
    # Validate hparams
    if not validate_hparams(hparam_all, hparam_keys["bc"], hparam_keys["hybrid"]):
        return 1
    
    # Check for existing BC checkpoint
    existing_bc = find_best_bc_checkpoint(args.dataset, args.env, args.names)
    bc_checkpoint = None
    bc_val_loss = None
    bc_epoch = None
    
    if existing_bc and args.skip_bc:
        bc_checkpoint, bc_val_loss, bc_epoch = existing_bc
        print(f"Skipping BC training; using existing checkpoint")
    elif existing_bc and not args.force_bc_retrain:
        response = input(
            f"\nFound existing BC checkpoint: {existing_bc[0].name}\n"
            f"val_loss: {existing_bc[1]:.6f}\n"
            f"Retrain? [y/N]: "
        )
        if response.lower() == 'y':
            bc_checkpoint = run_bc_training(args, hparam_keys["bc"])
            if bc_checkpoint:
                result = find_best_bc_checkpoint(args.dataset, args.env)
                if result:
                    bc_checkpoint, bc_val_loss, bc_epoch = result
        else:
            bc_checkpoint, bc_val_loss, bc_epoch = existing_bc
            print(f"Using existing BC checkpoint")
    else:
        # Train new BC
        bc_checkpoint = run_bc_training(args, hparam_keys["bc"])
        if bc_checkpoint:
            result = find_best_bc_checkpoint(args.dataset, args.env)
            if result:
                bc_checkpoint, bc_val_loss, bc_epoch = result
    
    if not bc_checkpoint:
        print(f"\nPipeline failed: No BC checkpoint available")
        return 1
    
    # Train hybrid PPO
    hybrid_save_file = run_hybrid_ppo_training(
        args,
        hparam_keys["hybrid"],
        bc_checkpoint,
        bc_val_loss
    )
    
    success = hybrid_save_file is not None
    
    # Log pipeline run
    log_pipeline_run(
        args,
        hparam_keys,
        bc_checkpoint,
        bc_val_loss,
        bc_epoch,
        hybrid_save_file,
        success
    )
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print(f"PIPELINE COMPLETED SUCCESSFULLY")
        print(f"   BC checkpoint: {bc_checkpoint.name}")
        print(f"   Hybrid save file: {hybrid_save_file}")
        print(f"   See logs/pipeline_runs.json for details")
    else:
        print(f"PIPELINE FAILED at hybrid PPO training stage")
    print(f"{'='*60}\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
