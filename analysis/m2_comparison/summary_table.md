# M2 Baseline — Model Comparison

| Model | Params | EX ↑ | Invalid SQL ↓ | Avg Latency ↓ | Peak VRAM ↓ |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 1.5B | 0.4500 | 32.0% | 2409 ms | 1.18 GB |
| Qwen2.5-Coder-3B | 3B | 0.5900 | 21.0% | 3687 ms | 2.06 GB |
| DeepSeek-Coder-6.7B | 6.7B | 0.6300 | 4.0% | 6250 ms | 4.43 GB |
| Qwen2.5-Coder-7B | 7B | 0.7900 | 2.0% | 4776 ms | 5.43 GB |

# Model Selection — Qwen2.5-Coder-3B
1. **High learning delta**: baseline EX of 59.0% and 21.0% Invalid SQL provide substantial headroom for SFT and RL to demonstrate a dramatic post-training improvement curve.
2. **Resource & cost efficiency**: lightweight memory footprint (2.06 GB peak VRAM in inference, ~6 GB in training) minimizes credit spend on Lightning AI and ensures safe, fast iterations without OOM risk.
3. **GRPO feasibility on budget**: enables rapid rollout generation with $G \ge 4$ candidates on Lightning L4 well within the ~$15 credit budget.
4. **Decision**: choose Qwen2.5-Coder-3B to maximize the demonstrable improvement signal and keep post-training iteration cycles fast and economical under hardware constraints.

**Trade-off**: requires SFT to significantly reduce the 21% invalid SQL rate before GRPO exploration; lower absolute accuracy ceiling than the 7B model.