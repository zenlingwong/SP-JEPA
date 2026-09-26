"""Stable command entry points for the comparison methods."""

import argparse
import os
import runpy
import sys
from pathlib import Path

from dependencies import verify_files


BASELINES = {
    "unet": "unet",
    "convlstm": "convlstm",
    "fno": "fno",
    "pde_transformer_mse": "pde_transformer_mse",
    "neuralom": "neuralom",
    "pde_transformer_pretrained": "pde_transformer_pretrained",
}
EXTERNAL_METHODS = {
    "pde_transformer_mse", "dreamerv3", "lewm", "eawm", "tc_lewm",
    "dpot", "poseidon", "pde_transformer_pretrained", "enma",
}


def launch(method, task="train"):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dependency-root", "--cache-root", dest="dependency_root", type=Path)
    if method in {"dpot", "poseidon", "climax"}:
        parser.add_argument("--variant", choices=("pretrained", "scratch"), required=True)
    options, remaining = parser.parse_known_args()
    def supplied(flag):
        return any(arg == flag or arg.startswith(flag + "=") for arg in remaining)

    if supplied("--method"):
        parser.error("method is fixed by this directory")
    if method in {"lewm", "eawm", "tc_lewm"} and (supplied("--event-weight") or supplied("--sigreg-temporal-window")):
        parser.error("event and temporal regularizer settings are fixed by this method")
    if method in EXTERNAL_METHODS:
        try:
            external = verify_files(method, options.dependency_root)
        except (FileNotFoundError, ValueError) as error:
            parser.error(f"{error}; prepare the dependency cache with --archive or --archive-url first")
        os.environ["OCEAN_COMPARISON_EXTERNAL_CODE"] = str(external)
        sys.path.insert(0, str(external / "vendor"))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if task == "evaluate":
        module = "evaluate"
        prefix = ["--method", method]
        if method in {"dpot", "poseidon", "climax"}:
            prefix += ["--variant", options.variant]
    elif method in BASELINES:
        module = "train_baseline"
        prefix = ["--method", BASELINES[method]]
    elif method in {"dpot", "poseidon"}:
        module = "train_baseline"
        prefix = ["--method", f"{method}_{options.variant}"]
    elif method == "climax":
        if task == "train":
            present = supplied("--pretrained-path")
            if (options.variant == "pretrained") != present:
                parser.error("ClimaX pretrained variant requires --pretrained-path; scratch variant omits it")
        module = "train_climax"
        prefix = []
    elif method == "dreamerv3":
        module = "train_dreamerv3"
        prefix = []
    elif method in {"lewm", "eawm", "tc_lewm"}:
        module = "train_worldmodel"
        prefix = {"lewm": [], "eawm": ["--event-weight", "0.1"], "tc_lewm": ["--sigreg-temporal-window", "4"]}[method]
    elif method == "enma":
        module = "train_enma"
        prefix = []
    else:
        raise ValueError(method)
    sys.argv = [module, *prefix, *remaining]
    runpy.run_module(module, run_name="__main__")
