"""
src/training/sft.py
===================
Core SFT training logic. Called by scripts/run_sft.py.
"""

import json
import random

import torch
from datasets import Dataset


def load_and_split(path: str, n_train: int, n_eval: int, seed: int) -> tuple[Dataset, Dataset]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    assert len(rows) >= n_train + n_eval, f"Need >= {n_train + n_eval} rows, got {len(rows)}"

    rng = random.Random(seed)
    rng.shuffle(rows)
    return Dataset.from_list(rows[:n_train]), Dataset.from_list(rows[n_train : n_train + n_eval])


def apply_chat_template(dataset: Dataset, tokenizer) -> Dataset:
    def fmt(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
        }
    return dataset.map(fmt, batched=True)


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

    train_data, eval_data = load_and_split(
        data_cfg["path"], data_cfg["n_train"], data_cfg["n_eval"], seed=training_cfg["seed"]
    )
    train_data = apply_chat_template(train_data, tokenizer)
    eval_data  = apply_chat_template(eval_data,  tokenizer)

    epoch_args = (
        {"max_steps": max_steps}
        if max_steps > 0
        else {"num_train_epochs": training_cfg["epochs"]}
    )

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
        eval_strategy="epoch" if max_steps <= 0 else "no",
        save_strategy="epoch" if max_steps <= 0 else "no",
        load_best_model_at_end=max_steps <= 0,
        metric_for_best_model="eval_loss",
        output_dir=output_cfg["dir"],
        logging_steps=10,
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

    eff_batch = training_cfg["batch_size"] * training_cfg["grad_accum"]
    print(f"Model: {model_cfg['name']}  |  r={lora_cfg['r']}  |  epochs={training_cfg['epochs']}  |  eff_batch={eff_batch}")
    trainer.train()

    if max_steps <= 0:
        model.save_pretrained(output_cfg["dir"])
        tokenizer.save_pretrained(output_cfg["dir"])
        print(f"Adapter saved: {output_cfg['dir']}")
