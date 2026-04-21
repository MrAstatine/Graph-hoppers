# Paradigm A – GNN→DRL on Cora

**Project**: Undergraduate benchmark comparing GNN→DRL vs. DRL→GNN on the Cora citation graph.  
**This folder**: Paradigm A — a GCN encodes the Cora graph; a PPO policy acts on GCN-derived observations.

---

## File Map

| File | Role |
|---|---|
| `gcn_encoder.py` | GCNEncoder + GCNClassifier (replaces `mpnn.py`) |
| `cora_env.py` | Gymnasium env for RL on Cora (replaces `environment1.py`) |
| `train_baseline.py` | Phase II – standalone GCN node-classification baseline |
| `train_paradigmA.py` | Phase III – PPO agent using frozen GCN encoder |
| `evaluate_paradigmA.py` | Phase V – side-by-side eval of GCN vs PPO (clean + OOD) |
| `requirements.txt` | Python dependencies |

---

## Colab Quick-Start

```python
# Cell 1 – Install deps
!pip install torch-geometric stable-baselines3[extra] wandb -q

# Cell 2 – Clone your fork
!git clone https://github.com/MrAstatine/Graph-hoppers
%cd graph-hoppers/paradigmA_cora

# Cell 3 – Train GCN baseline (Phase II)
!python train_baseline.py --epochs 200 --seed 42

# Cell 4 – Train Paradigm A PPO (Phase III)
!python train_paradigmA.py \
    --gcn_ckpt models/gcn_baseline_seed42.pt \
    --timesteps 100000 \
    --seed 42

# Cell 5 – Evaluate (Phase V)
!python evaluate_paradigmA.py \
    --gcn_ckpt models/gcn_baseline_seed42.pt \
    --ppo_ckpt models/paradigmA_ppo_seed42/ppo_final \
    --perturb_ratio 0.15 \
    --seed 42
```

---

## Three-Seed Runs (for t-tests in the paper)

```bash
for SEED in 12 5 25; do
  python train_baseline.py    --seed $SEED
  python train_paradigmA.py   --gcn_ckpt models/gcn_baseline_seed${SEED}.pt --seed $SEED
done
```

---

## Key Metrics Produced

| Metric | Where it comes from |
|---|---|
| Clean test accuracy (GCN) | `train_baseline.py` console output |
| OOD accuracy drop (GCN) | `evaluate_paradigmA.py` results CSV |
| PPO episode reward curve | `logs/paradigmA_ppo_seed*.csv` |
| PPO OOD reward drop | `evaluate_paradigmA.py` results CSV |
| Train latency (min/epoch) | `logs/gcn_baseline_seed*.csv` |

---

## What Was Removed from the Original DRL-GNN Repo

The original repo (`knowledgedefinednetworking/DRL-GNN`) was built for
Optical Transport Network routing. The following were completely removed
and should **not** be re-introduced:

- `GraphEnv-v1` (OTN routing env)
- Topology generators: NSFNET, GEANT2, GBN
- Demand allocation / link capacity logic
- K-shortest-path enumeration
- `SAPAgent`, `LBAgent`
- Reward based on bandwidth/demand
- TensorFlow (replaced with PyTorch + PyG)

---

## Architecture Summary

```
Cora graph (2708 nodes, 1433 features each)
        │
        ▼
  GCNEncoder (3 GCNConv layers → 32-dim embeddings)
        │
   ┌────┴───────────────────┐
   │                        │
mean-pool                node emb[target]
(graph embedding)        (per-step target)
   │                        │
   └────────┬───────────────┘
            │  concatenate  → obs (64-dim)
            ▼
       PPO MlpPolicy
            │
            ▼
      action: which node to classify next
            │
            ▼
  reward = +1 (correct) / -1 (wrong) + entropy bonus
```
