"""
sft_smoke_test.py
=================
Validates the SFT setup before running on Lightning AI.

CPU checks (always run):
  - Config keys, batch sizes
  - Dataset split: sizes, no overlap, no duplicates within train
  - Chat template formatting + ChatML markers (train_on_responses_only compatibility)
  - Token length spot-check on 20 examples
  - Environment variables and output directory

GPU smoke test (--smoke, run on Lightning AI):
  - 2 training steps on 4 examples to verify the full Unsloth pipeline

Usage:
    python scripts/sft_smoke_test.py             # CPU checks only
    python scripts/sft_smoke_test.py --smoke     # + full pipeline smoke test (needs GPU)
"""

import argparse
import copy
import json
import os
import random
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.training.sft import load_and_split, train

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {PASS if ok else FAIL}  {label}" + (f"  — {detail}" if detail else ""))
    return ok


def load_rows(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def smoke_test(cfg: dict, train_rows: list) -> bool:
    """2-step training run on 4 examples. Reuses train() with overridden config."""
    try:
        from unsloth import FastLanguageModel  # noqa: F401 — check availability
    except ImportError as e:
        print(f"  {SKIP}  unsloth not available ({e}) — run on Lightning AI")
        return True

    smoke_dir = "checkpoints/smoke_test"
    try:
        os.makedirs(smoke_dir, exist_ok=True)
        tmp_path = os.path.join(smoke_dir, "smoke_data.jsonl")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in random.sample(train_rows, 4):
                f.write(json.dumps(row) + "\n")

        smoke_cfg = copy.deepcopy(cfg)
        smoke_cfg["data"]["path"]    = tmp_path
        smoke_cfg["data"]["n_train"] = 4
        smoke_cfg["data"]["n_eval"]  = 0
        smoke_cfg["output"]["dir"]   = smoke_dir

        train(smoke_cfg, max_steps=2, report_to="none")
        return True

    except Exception as e:
        print(f"  {FAIL}  smoke test crashed: {e}")
        return False
    finally:
        if os.path.exists(smoke_dir):
            shutil.rmtree(smoke_dir)


# ── CPU checks ────────────────────────────────────────────────────────────────

def check_config(cfg: dict) -> bool:
    all_ok = True
    required = {
        "model":    ["name", "max_seq_length", "load_in_4bit"],
        "lora":     ["r", "alpha", "dropout", "bias", "target_modules"],
        "training": ["epochs", "batch_size", "grad_accum", "learning_rate",
                     "weight_decay", "lr_scheduler", "warmup_ratio", "optim", "seed"],
        "data":     ["path", "n_train", "n_eval"],
        "output":   ["dir"],
        "wandb":    ["project", "name"],
    }
    for section, keys in required.items():
        missing = [k for k in keys if k not in cfg.get(section, {})]
        all_ok &= check(f"{section} keys", not missing,
                        f"missing: {missing}" if missing else "")

    eff = cfg["training"]["batch_size"] * cfg["training"]["grad_accum"]
    all_ok &= check(f"Effective batch = {eff}", eff >= 8)
    return all_ok


def check_dataset(cfg: dict) -> tuple[bool, list, list]:
    all_ok = True
    data_cfg     = cfg["data"]
    training_cfg = cfg["training"]

    path = data_cfg["path"]
    all_ok &= check(f"{path} exists", os.path.exists(path))
    if not os.path.exists(path):
        return all_ok, [], []

    rows = load_rows(path)
    n_train, n_eval = data_cfg["n_train"], data_cfg["n_eval"]
    all_ok &= check(f"Row count >= {n_train + n_eval}", len(rows) >= n_train + n_eval,
                    f"got {len(rows)}")

    train_ds, eval_ds = load_and_split(path, n_train, n_eval, seed=training_cfg["seed"])
    train_rows = train_ds.to_list()
    eval_rows  = eval_ds.to_list()

    all_ok &= check(f"Train size = {n_train}", len(train_rows) == n_train)
    all_ok &= check(f"Eval size  = {n_eval}",  len(eval_rows)  == n_eval)

    train_keys = [r["messages"][1]["content"] for r in train_rows]
    eval_keys  = [r["messages"][1]["content"] for r in eval_rows]
    all_ok &= check("No train/eval overlap",
                    len(set(train_keys) & set(eval_keys)) == 0)
    dupes = len(train_keys) - len(set(train_keys))
    all_ok &= check("No duplicates within train", dupes == 0,
                    f"{dupes} — re-run curate_sft_dataset.py" if dupes else "")

    train_ds2, _ = load_and_split(path, n_train, n_eval, seed=training_cfg["seed"])
    all_ok &= check("Split is deterministic",
                    train_rows[0]["messages"][-1]["content"] ==
                    train_ds2[0]["messages"][-1]["content"])

    return all_ok, train_rows, eval_rows


def check_chat_template(cfg: dict, train_rows: list, eval_rows: list) -> bool:
    all_ok = True
    print("   (uses cached tokenizer after first download)")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        all_ok &= check(f"Tokenizer loaded: {cfg['model']['name']}", True)
    except Exception as e:
        check("Tokenizer loaded", False, str(e))
        return False

    sample = train_rows[0]["messages"]
    try:
        formatted = tokenizer.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=False
        )
        all_ok &= check("apply_chat_template succeeds", True)
    except Exception as e:
        all_ok &= check("apply_chat_template succeeds", False, str(e))
        return all_ok

    all_ok &= check("Instruction marker: '<|im_start|>user\\n'",
                    "<|im_start|>user\n" in formatted)
    all_ok &= check("Response marker:    '<|im_start|>assistant\\n'",
                    "<|im_start|>assistant\n" in formatted)
    all_ok &= check("Gold SQL in output", sample[-1]["content"] in formatted)
    all_ok &= check("Ends with <|im_end|>", formatted.strip().endswith("<|im_end|>"))

    max_seq = cfg["model"]["max_seq_length"]
    over = []
    for row in random.Random(0).sample(train_rows + eval_rows, 20):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        if len(tokenizer.encode(text)) > max_seq:
            over.append(len(tokenizer.encode(text)))
    all_ok &= check(f"20 samples within {max_seq} tokens", len(over) == 0,
                    f"{len(over)} over limit: {over}" if over else "")

    return all_ok


def check_environment(cfg: dict) -> bool:
    all_ok = True
    all_ok &= check("WANDB_API_KEY set", bool(os.environ.get("WANDB_API_KEY")),
                    "export WANDB_API_KEY=...")
    all_ok &= check("HF_TOKEN set",      bool(os.environ.get("HF_TOKEN")),
                    "export HF_TOKEN=...")
    try:
        os.makedirs(cfg["output"]["dir"], exist_ok=True)
        all_ok &= check(f"Output dir writable: {cfg['output']['dir']}", True)
    except Exception as e:
        all_ok &= check(f"Output dir writable: {cfg['output']['dir']}", False, str(e))
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--smoke",  action="store_true",
                        help="Run 2-step GPU training smoke test (requires GPU + unsloth)")
    args = parser.parse_args()

    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Cannot load config: {e}")
        sys.exit(1)

    all_ok = True
    print("\n=== SFT Smoke Test ===\n")

    print("1. Config:")
    all_ok &= check_config(cfg)

    print("\n2. Dataset:")
    dataset_ok, train_rows, eval_rows = check_dataset(cfg)
    all_ok &= dataset_ok

    print("\n3. Chat template:")
    if train_rows:
        all_ok &= check_chat_template(cfg, train_rows, eval_rows)
    else:
        print(f"  {SKIP}  dataset missing — fix step 2 first")

    print("\n4. Environment:")
    all_ok &= check_environment(cfg)

    t = cfg["training"]
    n_train = cfg["data"]["n_train"]
    steps_per_epoch = n_train // (t["batch_size"] * t["grad_accum"])
    print(f"\nTraining estimate: {steps_per_epoch} steps/epoch × {t['epochs']} epochs "
          f"= {steps_per_epoch * t['epochs']} total steps")

    if args.smoke:
        print("\n5. GPU smoke test (2 steps, 4 examples):")
        if train_rows:
            ok = smoke_test(cfg, train_rows)
            check("2-step training run", ok)
            all_ok &= ok
        else:
            print(f"  {SKIP}  dataset missing")

    print(f"\n{'=' * 38}")
    if all_ok:
        print("  ALL PASSED")
        if not args.smoke:
            print("  Run with --smoke on Lightning AI to verify the full pipeline")
    else:
        print("  SOME FAILED — fix before running on Lightning AI")
    print(f"{'=' * 38}\n")


if __name__ == "__main__":
    main()

