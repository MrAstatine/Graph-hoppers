# Project Setup with uv

This project uses [uv](https://github.com/astral-sh/uv) for Python package management and environment setup.

## Getting Started

1. **Install uv** (if not already installed):
	```sh
	pip install uv
	```

2. **Sync dependencies:**
	```sh
	uv sync
	```
	This will install all dependencies specified in `pyproject.toml`.


---

## Running the A_on_cora Experiments

The `A_on_cora` folder contains code for running Paradigm A (GCN → DRL) experiments on the Cora citation graph. Below are the main scripts and how to use them:

### Main Scripts

| Script                  | Purpose                                                      |
|-------------------------|--------------------------------------------------------------|
| `train_baseline.py`     | Train and evaluate the standalone GCN node-classification baseline |
| `train_paradigmA.py`    | Train the PPO agent using the frozen GCN encoder             |
| `evaluate_paradigmA.py` | Evaluate GCN and PPO agents (clean and OOD)                  |

### Example Commands

**Train GCN Baseline:**
```sh
cd A_on_cora
python train_baseline.py --epochs 200 --seed 12
```

**Train Paradigm A PPO Agent:**
```sh
python train_paradigmA.py --gcn_ckpt models/gcn_baseline_seed42.pt --timesteps 100000 --seed 12
```

**Evaluate (GCN vs PPO, Clean & OOD):**
```sh
python evaluate_paradigmA.py \
		--gcn_ckpt models/gcn_baseline_seed12.pt \
		--ppo_ckpt models/paradigmA_ppo_seed42/ppo_final \
		--perturb_ratio 0.15 \
		--seed 12
```

**Three-Seed Runs (for statistical testing):**
```sh
for SEED in 12 5 25; do
	python train_baseline.py    --seed $SEED
	python train_paradigmA.py   --gcn_ckpt models/gcn_baseline_seed${SEED}.pt --seed $SEED
done
```

**Outputs:**
- Model checkpoints are saved in `A_on_cora/models/`
- Logs and metrics are saved in `A_on_cora/logs/`
- Evaluation results are saved in `A_on_cora/results/`

For more details on the experiment design and file map, see `A_on_cora/README.md`.

---
For more details, see the our repo: [Project Repo](https://github.com/MrAstatine/Graph-hoppers)
