"""
src/training/sft.py
===================
Core SFT training logic. Called by scripts/run_sft.py.
"""

import json
import os
import random

import torch
from datasets import Dataset
from transformers import TrainerCallback


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_and_split(data_cfg: dict, seed: int) -> tuple[Dataset, Dataset]:
    """
    Load train/eval datasets from config.

    Supports two modes:
      - Separate files: data_cfg has 'train_path' and 'eval_path'
      - Single file:    data_cfg has 'path', 'n_train', 'n_eval' (legacy)
    """
    if "train_path" in data_cfg:
        train_rows = load_jsonl(data_cfg["train_path"])
        eval_rows  = load_jsonl(data_cfg["eval_path"])
    else:
        rows = load_jsonl(data_cfg["path"])
        n_train, n_eval = data_cfg["n_train"], data_cfg["n_eval"]
        assert len(rows) >= n_train + n_eval, \
            f"Need >= {n_train + n_eval} rows, got {len(rows)}"
        rng = random.Random(seed)
        rng.shuffle(rows)
        train_rows = rows[:n_train]
        eval_rows  = rows[n_train: n_train + n_eval]

    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def apply_chat_template(dataset: Dataset, tokenizer) -> Dataset:
    def fmt(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
        }
    return dataset.map(fmt, batched=True)


class EXCheckpointCallback(TrainerCallback):
    """Run EX eval once per epoch; save adapter to best_dir when EX improves."""

    def __init__(self, records: list[dict], tokenizer, best_dir: str):
        self.records   = records
        self.tokenizer = tokenizer
        self.best_dir  = best_dir
        self.best_ex   = 0.0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        from unsloth import FastLanguageModel
        from src.eval.evaluator import evaluate

        # generate SQL for each eval record
        FastLanguageModel.for_inference(model)
        predicted_sqls = []
        for record in self.records:
            msgs = [m for m in record["messages"] if m["role"] != "assistant"]
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            sql = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            predicted_sqls.append(sql)
        FastLanguageModel.for_training(model)

        # compute EX and log to W&B
        result = evaluate(self.records, predicted_sqls)
        ex = result["execution_accuracy"]
        if state.is_world_process_zero:
            import wandb
            wandb.log({"eval/execution_accuracy": ex}, step=state.global_step)
            print(f"  EX epoch {int(state.epoch)}: {ex:.4f}  (best: {self.best_ex:.4f})")

        # save adapter if new best EX
        if ex > self.best_ex:
            self.best_ex = ex
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            if state.is_world_process_zero:
                print(f"  → Saved best model to {self.best_dir}")

        return control


def train(cfg: dict, max_steps: int = -1, report_to: str = "wandb") -> None:
    """
    Run SFT training.

    Args:
        cfg:        Full config dict (from sft.yaml).
        max_steps:  If > 0, overrides num_train_epochs (used by smoke test).
        report_to:  Logging target. Pass "none" to disable W&B (used by smoke test).
    """

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    model_cfg    = cfg["model"]
    lora_cfg     = cfg["lora"]
    training_cfg = cfg["training"]
    data_cfg     = cfg["data"]
    output_cfg   = cfg["output"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=model_cfg["load_in_4bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # build EX eval records for mid-training callback (skipped in smoke tests)
    if max_steps <= 0:
        from src.data.loader import load_eval_set
        raw_ex    = load_jsonl(data_cfg["eval_path"])
        dev_path  = data_cfg.get("dev_set_path", "data/dev_set.jsonl")
        eval_recs = load_eval_set(dev_path, "data/")
        ex_records = [{**ev, "messages": r["messages"]} for r, ev in zip(raw_ex, eval_recs)]
        best_dir = os.path.join(output_cfg["dir"], "best_model")
        ex_cb = EXCheckpointCallback(ex_records, tokenizer, best_dir)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg["seed"],
        target_modules=lora_cfg["target_modules"],
    )

    train_data, eval_data = load_and_split(data_cfg, seed=training_cfg["seed"])
    train_data = apply_chat_template(train_data, tokenizer)
    eval_data  = apply_chat_template(eval_data,  tokenizer)

    epoch_args = (
        {"max_steps": max_steps}
        if max_steps > 0
        else {"num_train_epochs": training_cfg["epochs"]}
    )

    eval_steps    = training_cfg.get("eval_steps", None)
    use_steps     = eval_steps is not None and max_steps <= 0
    eval_strategy = "steps" if use_steps else ("epoch" if max_steps <= 0 else "no")
    save_strategy = eval_strategy if max_steps <= 0 else "no"

    sft_config=SFTConfig(
        **epoch_args,
        per_device_train_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["grad_accum"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        max_seq_length=model_cfg["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=2,
        lr_scheduler_type=training_cfg["lr_scheduler"],
        warmup_ratio=training_cfg["warmup_ratio"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=training_cfg["optim"],
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps if use_steps else None,
        save_steps=eval_steps if use_steps else None,
        load_best_model_at_end=max_steps <= 0,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        output_dir=output_cfg["dir"],
        logging_steps=20,
        report_to=report_to,
        seed=training_cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data if max_steps <= 0 else None,
        args=sft_config,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    if max_steps <= 0:
        trainer.add_callback(ex_cb)

    eff_batch = training_cfg["batch_size"] * training_cfg["grad_accum"]
    print(f"Model: {model_cfg['name']}  |  r={lora_cfg['r']}  |  epochs={training_cfg['epochs']}  |  eff_batch={eff_batch}")
    trainer.train()

    if max_steps <= 0:
        model.save_pretrained(output_cfg["dir"])
        tokenizer.save_pretrained(output_cfg["dir"])
        print(f"Adapter saved: {output_cfg['dir']}")

