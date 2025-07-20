from Scripts.train_val_test import train_validate_test
from Scripts.red_inference import red_sd
from Scripts.tikhonov_restore import tikhonov_restore
from Scripts.TV_restore import tv_restore
import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parents[1]))