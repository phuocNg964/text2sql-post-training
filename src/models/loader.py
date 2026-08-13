"""
Model loader for Text-to-SQL inference.

Extracted from smoke_test.py for reuse across scripts.
Always uses 4-bit quantization (NF4) — required for 7B models on T4 (16GB VRAM).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(
    model_name: str,
    load_in_4bit: bool = True,
):
    """
    Load a causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g. "Qwen/Qwen2.5-Coder-7B-Instruct")
        load_in_4bit: if True, load with NF4 4-bit quantization (default: True)

    Returns:
        (model, tokenizer) tuple. Model is in eval mode on the best available device.
    """
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
    return model, tokenizer
