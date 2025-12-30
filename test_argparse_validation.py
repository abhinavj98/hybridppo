
import sys
sys.path.insert(0, "training_files")
from train_hybrid_pipeline import validate_hparams
import yaml

with open("hparam.yml") as f:
    hparam_all = yaml.safe_load(f)

# Verify the hparams exist
result = validate_hparams(hparam_all, "Walker2d-v4-bc-large", "Walker2d-v4-hybrid")
if result:
    print("PASSED: Hparams validated successfully")
else:
    print("FAILED: Hparam validation failed")
    sys.exit(1)
