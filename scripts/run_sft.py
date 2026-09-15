"""
run_sft.py — SFT entry point.
Config: configs/sft.yaml | Run on: Lightning AI T4
"""

import argparse
import os
import sys

import wandb
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()
from src.training.sft import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wandb_cfg = cfg["wandb"]
    wandb.init(project=wandb_cfg["project"], name=wandb_cfg["name"],
               tags=wandb_cfg.get("tags", []), config=cfg)

    train(cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
