"""
Model loader for Text-to-SQL inference.

Small models (≤3B params, ~6 GB float16) load in float16.
Large models (>3B) auto-switch to 4-bit NF4 via bitsandbytes to fit on a T4
without CPU offloading. Quality impact on eval is negligible for baselines.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_3B = 3_000_000_000


def _use_4bit(model_name: str) -> bool:
    """Return True if the model is too large for float16 on the available GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        n_params = getattr(cfg, "num_parameters", None)
        # Fall back to hidden/layers estimate when num_parameters is absent
        if n_params is None:
            h = getattr(cfg, "hidden_size", 4096)
            l = getattr(cfg, "num_hidden_layers", 32)
            n_params = h * h * l * 12  # rough transformer param estimate
        return n_params > _3B
    except Exception:
        return False  # unknown model — let device_map handle it


def load_model(model_name: str, adapter: str | None = None):
    """
    Load a causal LM and tokenizer for inference.
    Automatically uses 4-bit NF4 for models >3B to avoid CPU offloading on T4.

    Returns:
        (model, tokenizer) tuple in eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if _use_4bit(model_name):
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading in 4-bit NF4 (model >3B)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )

    if adapter:
        from peft import PeftModel
        print(f"Merging adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer
