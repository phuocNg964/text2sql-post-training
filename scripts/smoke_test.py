"""
M0 Smoke Test
=============
Verifies the full stack is operational before any real training:
  1. All major dependencies import without error
  2. Qwen2.5-Coder-7B-Instruct loads in 4-bit quantization
  3. Model generates a SQL query from a sample prompt
  4. VRAM usage is logged to W&B
  5. W&B run finishes cleanly

Run:
    python scripts/smoke_test.py

Expected outcome: no exceptions, W&B run visible at wandb.ai
"""

import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from dotenv import load_dotenv

load_dotenv()


def check_imports() -> None:
    print("[ 1/5 ] Checking imports...")
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import peft  # noqa: F401
    import trl  # noqa: F401
    import datasets  # noqa: F401
    import sqlglot  # noqa: F401
    import wandb  # noqa: F401
    print("        All imports OK")


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = cfg["model"]["name"]
    load_in_4bit = cfg["model"].get("load_in_4bit", True)

    print(f"[ 2/5 ] Loading {model_name} ({'4-bit' if load_in_4bit else 'full'})...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if load_in_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("        Model loaded OK")
    return model, tokenizer


def run_inference(model, tokenizer) -> str:
    import torch

    print("[ 3/5 ] Running inference on sample prompt...")

    prompt = (
        "Given the schema:\n"
        "CREATE TABLE employees (id INTEGER, name TEXT, salary REAL, dept_id INTEGER);\n"
        "CREATE TABLE departments (id INTEGER, name TEXT);\n\n"
        "Question: What is the average salary per department?\n\n"
        "SQL:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    print(f"        Generated SQL:\n        {generated}")
    return generated


def get_vram_usage() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                "vram/allocated_gb": round(allocated, 2),
                "vram/reserved_gb": round(reserved, 2),
                "vram/total_gb": round(total, 2),
            }
    except Exception:
        pass
    return {"vram/allocated_gb": -1, "vram/note": "no CUDA device"}


def log_to_wandb(cfg: dict, sql: str, vram: dict) -> None:
    from src.utils.logging import finish_run, init_run, log_metrics

    print("[ 4/5 ] Logging to W&B...")
    init_run(
        config=cfg,
        project=cfg["wandb"]["project"],
        name="smoke-test",
    )
    log_metrics({"smoke_test/passed": 1, "smoke_test/sql_len": len(sql), **vram})
    finish_run()
    print("        W&B run finished OK")


def main() -> None:
    print("=" * 50)
    print("  M0 Smoke Test")
    print("=" * 50)

    check_imports()

    cfg = load_config()
    model, tokenizer = load_model(cfg)
    sql = run_inference(model, tokenizer)

    vram = get_vram_usage()
    print(f"[ 4/5 ] VRAM: {vram}")

    log_to_wandb(cfg, sql, vram)

    print()
    print("[ 5/5 ] All checks passed ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
