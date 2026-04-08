"""
CLI training script — run from project root:

    python notebooks/train.py
"""

import sys
import os

# Ensure project root is on path regardless of cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.trainer import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline(verbose=True)
