"""
train_baseline.py
-----------------
Phase II: Train and evaluate the standalone GCN baseline on Cora.

Usage (Colab / local):
    python train_baseline.py [--epochs 200] [--lr 0.01] [--seed 42]

Outputs
-------
  - models/gcn_baseline_seed{seed}.pt   (best checkpoint)
  - logs/gcn_baseline_seed{seed}.csv    (per-epoch metrics)
  - Prints final clean accuracy + OOD accuracy drop.
"""

import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import dropout_edge, add_random_edge

from gcn_encoder import GCNClassifier

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="GCN baseline on Cora")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perturb_ratio", type=float, default=0.15,
                   help="Edge perturbation ratio for OOD eval")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)


def perturb_graph(edge_index, ratio, num_nodes):
    """Drop + add random edges for OOD evaluation."""
    ei, _ = dropout_edge(edge_index, p=ratio, force_undirected=True)
    ei, _ = add_random_edge(ei, p=ratio, num_nodes=num_nodes, force_undirected=True)
    return ei


def accuracy(logits, labels, mask):
    pred = logits[mask].argmax(dim=1)
    correct = (pred == labels[mask]).sum().item()
    return correct / mask.sum().item()


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, perturb_ratio=0.0):
    model.eval()
    out = model(data.x, data.edge_index)
    val_acc = accuracy(out, data.y, data.val_mask)
    test_acc = accuracy(out, data.y, data.test_mask)

    ood_acc = None
    if perturb_ratio > 0:
        ei_perturbed = perturb_graph(data.edge_index, perturb_ratio, data.num_nodes)
        out_ood = model(data.x, ei_perturbed)
        ood_acc = accuracy(out_ood, data.y, data.test_mask)

    return val_acc, test_acc, ood_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load Cora
    dataset = Planetoid(root="./data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)
    print(f"Cora: {data.num_nodes} nodes, {data.num_edges} edges, {dataset.num_classes} classes")

    # Model
    model = GCNClassifier(
        in_channels=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # CSV logging
    log_path = f"logs/gcn_baseline_seed{args.seed}.csv"
    csv_file = open(log_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "loss", "val_acc", "test_acc", "epoch_time_s"])

    best_val = 0.0
    best_ckpt = f"models/gcn_baseline_seed{args.seed}.pt"
    t0_total = time.time()

    print(f"\nTraining GCN baseline (seed={args.seed}) for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, data, optimizer)
        val_acc, test_acc, _ = evaluate(model, data, perturb_ratio=0.0)
        elapsed = time.time() - t0

        writer.writerow([epoch, f"{loss:.4f}", f"{val_acc:.4f}", f"{test_acc:.4f}", f"{elapsed:.3f}"])

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_ckpt)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:03d} | loss={loss:.4f} | val={val_acc:.4f} | test={test_acc:.4f}")

    csv_file.close()
    total_time = time.time() - t0_total

    # Final evaluation with best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, clean_test_acc, ood_acc = evaluate(model, data, perturb_ratio=args.perturb_ratio)

    print(f"\n{'='*50}")
    print(f"GCN Baseline Results (seed={args.seed})")
    print(f"  Clean Test Accuracy :  {clean_test_acc:.4f}")
    print(f"  OOD  Test Accuracy  :  {ood_acc:.4f}  (perturb={args.perturb_ratio})")
    print(f"  OOD  Accuracy Drop  :  {clean_test_acc - ood_acc:.4f}")
    print(f"  Total Train Time    :  {total_time:.1f}s")
    print(f"  Best checkpoint     :  {best_ckpt}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
