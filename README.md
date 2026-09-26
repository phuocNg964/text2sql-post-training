# Text-to-SQL Post-Training Pipeline

Post-training pipeline for Text-to-SQL: **Evaluation → SFT → Execution-based RL (GRPO)**

**Stack:** Qwen2.5-Coder-3B-Instruct · Spider (train/dev/test) · W&B

---

## Setup

**1. Clone and install**
```bash
git clone <repo>
cd text2SQL-post-training
pip install -e ".[dev]"
```

**2. Set environment variables**
```bash
cp .env.example .env
# Fill in WANDB_API_KEY and HF_TOKEN in .env
```

**3. Run smoke test** (requires GPU with ≥16GB VRAM)
```bash
python scripts/smoke_test.py
```
Pass criteria: no exception, W&B run appears at `wandb.ai/<your-entity>/text2sql-post-training`

---

## Project Structure

```
src/
├── data/       # Dataset loading, preprocessing         (M1)
├── eval/       # SQL executor, evaluator                (M1)
├── models/     # Model loading, inference               (M2+)
├── training/   # SFT trainer, GRPO trainer              (M3+)
└── utils/      # Logging (W&B wrapper), config helpers
configs/        # YAML configs (one per experiment)
scripts/        # Entry-point scripts
notebooks/      # Analysis, error analysis
tests/          # Unit tests
```

---

## Milestones

| Milestone | Status | Description |
|---|---|---|
| M0 Foundation | ✅ | Project setup, experiment tracking |
| M1 Data & Eval | ✅ | Evaluation pipeline (Spider dev) |
| M2 Baseline | ✅ | Benchmarked 4 models; selected Qwen2.5-Coder-3B (59.0% EX) |
| M3 SFT | ✅ | QLoRA r=16 → 78.0% EX on the frozen 100-example holdout ([report](docs/m3_sft_report.md)) |
| M4 RL | 🔲 | GRPO with execution-based reward |
| M5 Final Eval | 🔲 | Full comparison + technical report |

---

## M3: Supervised Fine-Tuning

**Results:** 59% → 78% EX on Spider dev (n=100) · −59% inference latency · [Adapter on HF Hub](https://huggingface.co/PhuocNg9604/qwen2.5-coder-3b-text2sql-sft) · **Next:** GRPO RL (M4)

### My approach

I fine-tuned Qwen2.5-Coder-3B-Instruct using supervised fine-tuning with 4-bit QLoRA.

| Component | Configuration |
|---|---|
| Base model | Qwen2.5-Coder-3B-Instruct |
| Model size | 3.09B parameters |
| Trainable parameters | 29.9M — 0.96% of the model |
| LoRA | Rank 16, alpha 32 |
| Target modules | Attention and MLP projections (`q,k,v,o,gate,up,down_proj`) |
| Context length | 2,048 tokens |
| Training data | 1,000 examples |
| Validation data | 100 examples |
| Epochs | 3 |
| Effective batch size | 16 |
| Optimizer | 8-bit AdamW |
| LR schedule | Cosine decay with 5% warmup |
| Precision | bf16 with 4-bit quantised base |
| Hardware | One Tesla T4 (16 GB, ~\$0.19/hr) |


### Training data

The Spider train split has 7,000+ examples — I used 1,100, deliberately curated rather than randomly sampled.

The base model (M2) was run on 100 held-out dev examples and every failure was classified by gold SQL structure. The training pool was then weighted to **oversample the exact patterns the model was getting wrong**:

| Pool | Base failures | Training examples | SFT remaining | Outcome |
|---|:---:|:---:|:---:|:---:|
| Single-table (no JOIN in gold) | 23 | 611 | 11 | ✅ −52% |
| Negation (`NOT IN` / `EXCEPT`) | 4 | 146 | 2 | ⚠️ −50% |
| Multi-hop (≥ 2 joins) | 5 | 116 | 4 | ⚠️ −20% |
| Aggregation / `GROUP BY` | 3 | 116 | 2 | ⚠️ −33% |

> Pools are defined by gold SQL structure using the same `classify()` function as [`curate_sft_dataset.py`](scripts/curate_sft_dataset.py). A query can match multiple pools (e.g. negation + multi-hop); counts use exclusive assignment with priority p1 > p0 > p2 > p3.

Loss was computed on the SQL completion only (`train_on_responses_only`) — the model never trained on the question or schema tokens.

### Training curves

| Train loss | Eval loss |
|:---:|:---:|
| ![Train loss](docs/assets/train-loss.png) | ![Eval loss](docs/assets/eval-loss.png) |

Eval loss tracks train loss across all 3 epochs — no overfitting on the 1,000-sample set.

### Results

| Metric | Base | SFT | Δ |
|---|:---:|:---:|:---:|
| Execution accuracy (EX) | 59% | **78%** | **+19 pp** |
| Invalid SQL rate | 21% | **11%** | −10 pp |


**Prediction change matrix** (same 100 examples):

|  | SFT correct | SFT wrong |
|---|:---:|:---:|
| **Base correct** | Still correct: 56 | Regressed: **3** |
| **Base wrong** | Fixed: **22** | Still wrong: 19 |

### What SFT fixed — and what remains

The pool table above (Training data section) is the direct evidence. **Single-table queries drove the improvement** — 611 training examples targeting this pattern reduced failures from 23 to 11 (−52%). The model had been adding spurious JOINs to queries that needed none; SFT corrected that behaviour.

The other three pools saw only marginal gains despite comparable training budgets (116 examples each). Negation and aggregation improved moderately; multi-hop barely moved. The common thread: for p1/p2/p3, the failure mode isn't just pattern exposure — the model must reason about *which keys to join on*, *which aggregate to apply*, or *how to negate a set correctly* for schemas it has never seen. Imitation of correct examples provides weaker signal for these than for the structural no-JOIN pattern.

The persistent failures in p2/p3 are the direct motivation for M4 (GRPO): execution-based reward will directly penalise wrong join keys, wrong aggregates, and hallucinated column names — something SFT alone cannot.  
→ Full labeled failure sets: [`analysis/errors_3b_base_labeled.md`](analysis/errors_3b_base_labeled.md) · [`analysis/errors_3b_sft_labeled.md`](analysis/errors_3b_sft_labeled.md)
