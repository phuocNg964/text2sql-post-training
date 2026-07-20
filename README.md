# Text-to-SQL Post-Training Pipeline

Post-training pipeline for Text-to-SQL: **Evaluation → SFT → Execution-based RL (GRPO)**

**Stack:** Qwen2.5-Coder-7B-Instruct · Spider (train/dev) · BIRD (held-out) · W&B

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
| M1 Data & Eval | 🔲 | Evaluation pipeline (Spider dev + BIRD) |
| M2 Baseline | 🔲 | Benchmark candidate models |
| M3 SFT | 🔲 | Supervised fine-tuning with LoRA |
| M4 RL | 🔲 | GRPO with execution-based reward |
| M5 Final Eval | 🔲 | Full comparison + technical report |
