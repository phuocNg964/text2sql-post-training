# M2 Baseline — Model Comparison

| Model | Params | EX ↑ | Invalid SQL ↓ | Avg Latency ↓ | Peak VRAM ↓ |
|---|---|---|---|---|---|
| qwen2.5_coder_1.5b_instruct | 1.5B | 0.1600 | 35.0% | 2614 ms | 1.18 GB |
| qwen2.5_coder_3b_fp16 | 3B | 0.6700 | 11.0% | 2297 ms | 5.88 GB |
| qwen2.5_coder_3b_instruct | 3B | 0.6700 | 14.0% | 3933 ms | 2.06 GB |
| qwen2.5_coder_3b_sft | 3B | 0.7500 | 9.0% | 2227 ms | 5.88 GB |
| deepseek_coder_6.7b_instruct | 6.7B | 0.6500 | 2.0% | 6939 ms | 4.43 GB |
| qwen2.5_coder_7b_instruct | 7B | 0.7700 | 5.0% | 5279 ms | 5.43 GB |