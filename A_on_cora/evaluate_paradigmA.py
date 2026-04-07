"""
evaluate_paradigmA.py
---------------------
Phase V: Evaluate a trained Paradigm A PPO agent on clean and OOD Cora.

Compares:
  - GCN Baseline (from train_baseline.py checkpoint)
  - Paradigm A PPO agent (from train_paradigmA.py checkpoint)

Outputs
-------
  - Prints a summary table (accuracy, OOD drop, latency)
  - Saves results/eval_results.csv

Usage:
    python evaluate_paradigmA.py \
        --gcn_ckpt models/gcn_baseline_seed42.pt \
        --ppo_ckpt models/paradigmA_ppo_seed42/ppo_final \
        [--seeds 42 0 1] [--perturb_ratio 0.15]
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import dropout_edge, add_random_edge

from gcn_encoder import GCNClassifier
from cora_env import CoraNodeClassEnv

os.makedirs("results", exist_ok=True)
DATA_ROOT = "./data"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gcn_ckpt", type=str, required=True)
    p.add_argument("--ppo_ckpt", type=str, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--perturb_ratio", type=float, default=0.15)
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GCN Baseline evaluation
# ---------------------------------------------------------------------------


def perturb_graph(edge_index, ratio, num_nodes):
    ei, _ = dropout_edge(edge_index, p=ratio, force_undirected=True)
    ei, _ = add_random_edge(ei, p=ratio, num_nodes=num_nodes, force_undirected=True)
    return ei


@torch.no_grad()
def eval_gcn(model, data, perturb_ratio=0.0):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[data.test_mask].argmax(dim=1)
    clean_acc = (pred == data.y[data.test_mask]).float().mean().item()

    ood_acc = None
    if perturb_ratio > 0:
        ei_p = perturb_graph(data.edge_index, perturb_ratio, data.num_nodes)
        out_ood = model(data.x, ei_p)
        pred_ood = out_ood[data.test_mask].argmax(dim=1)
        ood_acc = (pred_ood == data.y[data.test_mask]).float().mean().item()

    return clean_acc, ood_acc


# ---------------------------------------------------------------------------
# PPO Agent evaluation
# ---------------------------------------------------------------------------


def eval_ppo(agent, env_clean, env_ood, n_episodes=10):
    """
    Roll out PPO agent on clean and OOD environments.
    Returns mean episode reward and inferred accuracy proxy.
    """

    def rollout(env, n_ep):
        total_rewards = []
        for _ in range(n_ep):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                ep_reward += reward
                done = terminated or truncated
            total_rewards.append(ep_reward)
        return np.mean(total_rewards), np.std(total_rewards)

    clean_mean, clean_std = rollout(env_clean, n_episodes)
    ood_mean, ood_std = rollout(env_ood, n_episodes)
    return clean_mean, clean_std, ood_mean, ood_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = get_args()
    device = torch.device(args.device)

    dataset = Planetoid(root=DATA_ROOT, name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    # ------------------------------------------------------------------
    # Load GCN baseline
    # ------------------------------------------------------------------
    gcn_model = GCNClassifier(
        in_channels=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=dataset.num_classes,
    ).to(device)
    gcn_model.load_state_dict(torch.load(args.gcn_ckpt, map_location=device))

    t0 = time.time()
    gcn_clean, gcn_ood = eval_gcn(gcn_model, data, perturb_ratio=args.perturb_ratio)
    gcn_latency = (time.time() - t0) * 1000  # ms

    print(f"\nGCN Baseline")
    print(f"  Clean Test Acc : {gcn_clean:.4f}")
    print(f"  OOD  Test Acc  : {gcn_ood:.4f}")
    print(f"  OOD Drop       : {gcn_clean - gcn_ood:.4f}")
    print(f"  Latency (ms)   : {gcn_latency:.1f}")

    # ------------------------------------------------------------------
    # Load PPO agent
    # ------------------------------------------------------------------
    seed = args.seeds[0]  # use first seed for env construction

    env_clean = CoraNodeClassEnv(
        encoder=gcn_model.encoder,
        split="test",
        budget=args.budget,
        perturb_ratio=0.0,
        seed=seed,
        device=str(device),
    )
    env_clean.set_classifier(gcn_model.classifier)

    env_ood = CoraNodeClassEnv(
        encoder=gcn_model.encoder,
        split="test",
        budget=args.budget,
        perturb_ratio=args.perturb_ratio,
        seed=seed,
        device=str(device),
    )
    env_ood.set_classifier(gcn_model.classifier)

    ppo_agent = PPO.load(args.ppo_ckpt)

    t0 = time.time()
    ppo_clean_rew, ppo_clean_std, ppo_ood_rew, ppo_ood_std = eval_ppo(
        ppo_agent, env_clean, env_ood, n_episodes=20
    )
    ppo_latency = (time.time() - t0) * 1000

    print("\nParadigm A (GNN->DRL / PPO)")
    print(f"  Clean Ep Reward (mean±std) : {ppo_clean_rew:.3f} ± {ppo_clean_std:.3f}")
    print(f"  OOD   Ep Reward (mean±std) : {ppo_ood_rew:.3f} ± {ppo_ood_std:.3f}")
    print(f"  OOD Reward Drop            : {ppo_clean_rew - ppo_ood_rew:.3f}")
    print(f"  Latency (ms, 20 eps)       : {ppo_latency:.1f}")

    # ------------------------------------------------------------------
    # Save CSV summary
    # ------------------------------------------------------------------
    out_path = "results/eval_results" + seed + ".csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "clean_acc_or_rew",
                "ood_acc_or_rew",
                "ood_drop",
                "latency_ms",
                "perturb_ratio",
            ]
        )
        w.writerow(
            [
                "GCN_baseline",
                f"{gcn_clean:.4f}",
                f"{gcn_ood:.4f}",
                f"{gcn_clean - gcn_ood:.4f}",
                f"{gcn_latency:.1f}",
                args.perturb_ratio,
            ]
        )
        w.writerow(
            [
                "ParadigmA_PPO",
                f"{ppo_clean_rew:.4f}",
                f"{ppo_ood_rew:.4f}",
                f"{ppo_clean_rew - ppo_ood_rew:.4f}",
                f"{ppo_latency:.1f}",
                args.perturb_ratio,
            ]
        )

    print(f"\n[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()
