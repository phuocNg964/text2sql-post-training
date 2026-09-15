"""
Model loader for Text-to-SQL inference.

Loads in float16 (not 4-bit) — correct for inference since LoRA merging
at 4-bit precision introduces rounding errors. The 3B model fits in float16
on a T4 (16 GB) with ~6.4 GB VRAM.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str,
    adapter: str | None = None,
):
    """
    Load a causal LM and its tokenizer in float16 for inference.

    Args:
        model_name: HuggingFace model identifier or local path.
        adapter:    Optional LoRA adapter path or HF repo ID.
                    Merged into the base model via merge_and_unload().

    Returns:
        (model, tokenizer) tuple in eval mode on the best available device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter:
        from peft import PeftModel
        print(f"Merging adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer
