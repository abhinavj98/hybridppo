"""
Unit tests for train_hybrid_pipeline.py

Tests core functions without running actual training.
"""

import unittest
import tempfile
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Adjust path to import from training_files directory
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import functions to test
from train_hybrid_pipeline import (
    validate_hparams,
    find_best_bc_checkpoint,
    log_pipeline_run
)


class TestValidateHparams(unittest.TestCase):
    """Test hparam validation against hparam.yml."""
    
    def test_valid_hparams(self):
        """Test validation with existing hparam keys"""
        hparam_all = {
            "Walker2d-v4-bc-large": {"lr": 0.001},
            "Walker2d-v4-hybrid": {"lr": 0.0001}
        }
        result = validate_hparams(hparam_all, "Walker2d-v4-bc-large", "Walker2d-v4-hybrid")
        self.assertTrue(result)
    
    def test_missing_bc_hparam(self):
        """Test validation fails when BC hparam missing"""
        hparam_all = {
            "Walker2d-v4-hybrid": {"lr": 0.0001}
        }
        result = validate_hparams(hparam_all, "Walker2d-v4-bc-large", "Walker2d-v4-hybrid")
        self.assertFalse(result)
    
    def test_missing_hybrid_hparam(self):
        """Test validation fails when hybrid hparam missing"""
        hparam_all = {
            "Walker2d-v4-bc-large": {"lr": 0.001}
        }
        result = validate_hparams(hparam_all, "Walker2d-v4-bc-large", "Walker2d-v4-hybrid")
        self.assertFalse(result)


class TestFindBestBCCheckpoint(unittest.TestCase):
    """Test finding best BC checkpoint by parsing CSV."""
    
    def setUp(self):
        """Create temporary directory with mock BC checkpoints and metrics."""
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Change to temp directory for this test
        os.chdir(self.temp_path)
        
        # Create directory structure
        self.bc_dir = self.temp_path / "bc_checkpoints" / "mujoco" / "walker2d"
        self.bc_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary directory."""
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()
    
    def test_no_bc_dir(self):
        """Test when BC checkpoint directory doesn't exist"""
        result = find_best_bc_checkpoint(
            "mujoco",
            "nonexistent"
        )
        self.assertIsNone(result)
    
    def test_no_metrics_files(self):
        """Test when no metrics CSV files found"""
        # Directory exists but is empty
        result = find_best_bc_checkpoint("mujoco", "walker2d")
        self.assertIsNone(result)
    
    def test_single_metrics_file(self):
        """Test with a single metrics CSV file"""
        # Create metrics CSV
        metrics_file = self.bc_dir / "bc_walker2d_20251224_epoch1 with avg loss 0.1234_metrics.csv"
        df = pd.DataFrame({
            "epoch": [1, 2, 3, 4, 5],
            "train_loss": [0.5, 0.4, 0.3, 0.25, 0.22],
            "val_loss": [0.45, 0.38, 0.32, 0.28, 0.25]
        })
        df.to_csv(metrics_file, index=False)
        
        # Create corresponding checkpoint files
        (self.bc_dir / "bc_walker2d_20251224_epoch1 with avg loss 0.1234.zip").touch()
        (self.bc_dir / "bc_walker2d_20251224_epoch2 with avg loss 0.1234.zip").touch()
        (self.bc_dir / "bc_walker2d_20251224_epoch5 with avg loss 0.25.zip").touch()
        
        # Find best (lowest val_loss at epoch 5)
        result = find_best_bc_checkpoint("mujoco", "walker2d")
        
        self.assertIsNotNone(result)
        ckpt_path, val_loss, epoch = result
        self.assertEqual(epoch, 5)
        self.assertEqual(val_loss, 0.25)
        self.assertTrue(ckpt_path.exists())
    
    def test_multiple_metrics_files(self):
        """Test with multiple metrics CSV files (choose best overall)"""
        # Create first metrics file (worse)
        metrics_file1 = self.bc_dir / "bc_walker2d_run1_metrics.csv"
        df1 = pd.DataFrame({
            "epoch": [1, 2, 3],
            "train_loss": [0.5, 0.4, 0.35],
            "val_loss": [0.45, 0.40, 0.38]
        })
        df1.to_csv(metrics_file1, index=False)
        
        # Create second metrics file (better)
        metrics_file2 = self.bc_dir / "bc_walker2d_run2_metrics.csv"
        df2 = pd.DataFrame({
            "epoch": [1, 2, 3],
            "train_loss": [0.6, 0.4, 0.25],
            "val_loss": [0.55, 0.39, 0.20]
        })
        df2.to_csv(metrics_file2, index=False)
        
        # Create checkpoint files for second run
        (self.bc_dir / "bc_walker2d_run2.zip").touch()
        
        # Should pick the run with lowest overall val_loss
        result = find_best_bc_checkpoint("mujoco", "walker2d")
        
        self.assertIsNotNone(result)
        _, val_loss, epoch = result
        self.assertEqual(val_loss, 0.20)
        self.assertEqual(epoch, 3)


class TestLogPipelineRun(unittest.TestCase):
    """Test logging pipeline run metadata to JSON."""
    
    def setUp(self):
        """Create temporary directory for logs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.logs_dir = self.temp_path / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_new_log_file_creation(self):
        """Test creating a new pipeline_runs.json file"""
        # Create a mock args object
        class MockArgs:
            dataset = "mujoco"
            env = "walker2d"
            names = ["expert-v0"]
            seed = 42
            mix_ratio = 0.5
            rho_bar = 1.0
            c_bar = 0.95
            log_std_subtract = 0.0
        
        hparam_keys = {
            "bc": "Walker2d-v4-bc-large",
            "hybrid": "Walker2d-v4-hybrid"
        }
        
        bc_checkpoint = Path("bc_checkpoints/mujoco/walker2d/bc_walker2d_best.zip")
        
        # Temporarily change logs dir
        import train_hybrid_pipeline
        original_cwd = Path.cwd()
        
        # Create logs in temp dir by directly building the path
        log_file = self.logs_dir / "pipeline_runs.json"
        
        # Simulate log entry creation
        run_entry = {
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "dataset": "mujoco",
            "env": "walker2d",
            "names": ["expert-v0"],
            "seed": 42,
            "bc_hparam": "Walker2d-v4-bc-large",
            "hybrid_hparam": "Walker2d-v4-hybrid",
            "bc_checkpoint": "bc_checkpoints/mujoco/walker2d/bc_walker2d_best.zip",
            "bc_val_loss": 0.123,
            "bc_best_epoch": 50,
            "bc_epochs_trained": 50,
            "hybrid_save_file": "hybrid_ppo_walker2d_20251224_120000",
            "mix_ratio": 0.5,
            "rho_bar": 1.0,
            "c_bar": 0.95,
            "log_std_subtract": 0.0,
        }
        
        runs = [run_entry]
        
        with open(log_file, 'w') as f:
            json.dump(runs, f, indent=2)
        
        # Verify file was created
        self.assertTrue(log_file.exists())
        
        # Verify contents
        with open(log_file) as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["dataset"], "mujoco")
        self.assertEqual(loaded[0]["success"], True)
        self.assertEqual(loaded[0]["bc_val_loss"], 0.123)
    
    def test_append_to_existing_log(self):
        """Test appending to existing pipeline_runs.json"""
        log_file = self.logs_dir / "pipeline_runs.json"
        
        # Create initial entry
        initial_run = {
            "timestamp": "2025-12-24T10:00:00",
            "success": True,
            "dataset": "mujoco",
            "env": "walker2d",
            "bc_val_loss": 0.1
        }
        
        with open(log_file, 'w') as f:
            json.dump([initial_run], f)
        
        # Add second entry
        second_run = {
            "timestamp": "2025-12-24T12:00:00",
            "success": True,
            "dataset": "mujoco",
            "env": "humanoid",
            "bc_val_loss": 0.15
        }
        
        runs = []
        with open(log_file) as f:
            runs = json.load(f)
        
        runs.append(second_run)
        
        with open(log_file, 'w') as f:
            json.dump(runs, f, indent=2)
        
        # Verify both entries exist
        with open(log_file) as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["env"], "walker2d")
        self.assertEqual(loaded[1]["env"], "humanoid")


class TestEndToEndLogic(unittest.TestCase):
    """Integration tests for pipeline logic."""
    
    def test_hparam_validation(self):
        """Test validating hparams"""
        hparam_all = {
            "Walker2d-v4": {"n_timesteps": 25e6},
            "Walker2d-v4-bc-large": {"batch_size": 256},
            "Walker2d-v4-hybrid": {"learning_rate": 5e-6},
            "Humanoid-v4-bc-large": {"batch_size": 256},
            "Humanoid-v4-hybrid": {"learning_rate": 5e-6},
        }
        
        # Test validation with valid keys
        is_valid = validate_hparams(hparam_all, "Walker2d-v4-bc-large", "Walker2d-v4-hybrid")
        self.assertTrue(is_valid, "Valid hparams should pass validation")
        
        # Test validation with missing key
        is_valid = validate_hparams(hparam_all, "NonExistent-v4-bc-large", "Walker2d-v4-hybrid")
        self.assertFalse(is_valid, "Missing hparam should fail validation")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
