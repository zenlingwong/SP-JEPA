"""pde_transformer_mse train entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from entry import launch

if __name__ == "__main__":
    launch("pde_transformer_mse", "train")
