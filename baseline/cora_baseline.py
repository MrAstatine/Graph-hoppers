"""
Cora Baseline: Supervised GCN + GAT for Node Classification
============================================================
Project: DRL-GNN Paradigm Comparison (Phase I–II)
Purpose: Establish clean supervised reference before any DRL (Paradigm A/B).
Dataset: Planetoid Cora (~2,708 nodes, 5,429 edges, 7 classes, 1,433 features)
Hardware: Google Colab CPU or free T4 GPU (both work)

Install (run once in Colab):
    !pip install torch-geometric wandb

Usage:
    python cora_baseline.py               # GCN only
    python cora_baseline.py --model gat   # GAT only
    python cora_baseline.py --model both  # GCN then GAT
    python cora_baseline.py --ood         # add OOD perturbation eval
    python cora_baseline.py --wandb       # log to Weights & Biases

    # Both models + OOD + W&B
!python cora_baseline.py --model both --ood --wandb

The saved checkpoint (best_<model>.pt) is the reference for Paradigm A/B comparisons.
"""

import argparse
import time
import random
import os

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GATConv

# ── Optional W&B ─────────────────────────────────────────────────────────────
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────────────────


class GCN(torch.nn.Module):
    """Two-layer Graph Convolutional Network.

    Standard encoder used in Paradigm A (GNN → DRL) as the state embedder.
    Hidden dim 64 matches typical DRL-GNN configurations.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits; apply log_softmax outside for NLL loss


class GAT(torch.nn.Module):
    """Two-layer Graph Attention Network (optional second baseline).

    GAT provides attention-weighted aggregation; useful to check if
    attention mechanism matters before adding RL complexity.
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # out_channels heads → concat=False means average
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# OOD Perturbation
# ─────────────────────────────────────────────────────────────────────────────


def perturb_graph(data, edge_drop_rate=0.15, node_feat_noise=0.1, seed=42):
    """Return a perturbed copy of data for OOD evaluation.

    Two perturbations applied:
    - Random edge removal (structural): simulates missing citations.
    - Gaussian noise on node features (attribute): simulates noisy metadata.

    Parameters match the 10-20% perturbation range in rough1pager.docx.
    """
    rng = torch.Generator().manual_seed(seed)
    perturbed = data.clone()

    # --- Edge drop ---
    n_edges = perturbed.edge_index.size(1)
    keep_mask = torch.rand(n_edges, generator=rng) > edge_drop_rate
    perturbed.edge_index = perturbed.edge_index[:, keep_mask]
    edges_removed = int((~keep_mask).sum())

    # --- Feature noise ---
    noise = torch.randn_like(perturbed.x, generator=rng) * node_feat_noise
    perturbed.x = perturbed.x + noise

    print(
        f"  OOD perturbation: removed {edges_removed}/{n_edges} edges "
        f"({edge_drop_rate*100:.0f}%), added feature noise σ={node_feat_noise}"
    )
    return perturbed


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc


def run_training(model_name, data, args, device, use_wandb=False):
    """Full train–val–test cycle for one model on clean Cora."""

    # ── Build model ──────────────────────────────────────────────────────────
    num_features = data.num_node_features
    num_classes = data.y.max().item() + 1

    if model_name == "gcn":
        model = GCN(
            num_features, hidden_channels=64, out_channels=num_classes, dropout=0.5
        ).to(device)
    elif model_name == "gat":
        model = GAT(
            num_features,
            hidden_channels=8,
            out_channels=num_classes,
            heads=8,
            dropout=0.6,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    print(f"\n{'='*55}")
    print(
        f"  Training {model_name.upper()}  |  lr={args.lr}  |  "
        f"epochs={args.epochs}  |  seed={args.seed}"
    )
    print(f"{'='*55}")

    best_val_acc = 0.0
    best_epoch = 0
    patience_count = 0
    checkpoint_path = f"best_{model_name}.pt"

    epoch_times = []
    train_losses = []
    val_accs = []

    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = train_epoch(model, data, optimizer)
        epoch_ms = (time.perf_counter() - t0) * 1000

        val_acc = evaluate(model, data, data.val_mask)
        train_acc = evaluate(model, data, data.train_mask)

        epoch_times.append(epoch_ms)
        train_losses.append(loss)
        val_accs.append(val_acc)

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    f"{model_name}/train_loss": loss,
                    f"{model_name}/train_acc": train_acc,
                    f"{model_name}/val_acc": val_acc,
                    f"{model_name}/epoch_ms": epoch_ms,
                    "epoch": epoch,
                }
            )

        # ── Best checkpoint ──────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1

        # ── Logging ──────────────────────────────────────────────────────────
        if epoch % 20 == 0 or epoch == 1:
            avg_ms = sum(epoch_times[-20:]) / len(epoch_times[-20:])
            print(
                f"  Epoch {epoch:4d}  loss={loss:.4f}  "
                f"val={val_acc:.4f}  best_val={best_val_acc:.4f}  "
                f"avg_ms/epoch={avg_ms:.1f}"
            )

        # ── Early stopping ───────────────────────────────────────────────────
        if patience_count >= args.patience:
            print(
                f"  Early stopping at epoch {epoch} "
                f"(no val improvement for {args.patience} epochs)"
            )
            break

    total_time_s = time.perf_counter() - total_start

    # ── Final test evaluation (load best checkpoint) ─────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_acc = evaluate(model, data, data.test_mask)
    avg_epoch = sum(epoch_times) / len(epoch_times)

    print(f"\n  ── Clean Results ({model_name.upper()}) ──")
    print(f"  Best val acc  : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Test acc      : {test_acc:.4f}")
    print(f"  Total time    : {total_time_s:.1f}s")
    print(f"  Avg ms/epoch  : {avg_epoch:.1f} ms")
    print(f"  Checkpoint    : {checkpoint_path}")

    results = {
        "model": model_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "total_time_s": total_time_s,
        "avg_epoch_ms": avg_epoch,
        "best_epoch": best_epoch,
        "epochs_run": len(epoch_times),
    }

    # ── OOD evaluation ───────────────────────────────────────────────────────
    if args.ood:
        print(f"\n  ── OOD Evaluation ({model_name.upper()}) ──")
        ood_data = perturb_graph(
            data, edge_drop_rate=args.ood_edge_drop, node_feat_noise=args.ood_feat_noise
        )
        ood_data = ood_data.to(device)
        ood_test_acc = evaluate(model, ood_data, ood_data.test_mask)
        acc_drop = test_acc - ood_test_acc
        print(f"  OOD test acc  : {ood_test_acc:.4f}")
        print(
            f"  Accuracy drop : {acc_drop:.4f}  "
            f"({acc_drop/test_acc*100:.1f}% relative drop)"
        )
        results["ood_test_acc"] = ood_test_acc
        results["ood_acc_drop"] = acc_drop

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    f"{model_name}/ood_test_acc": ood_test_acc,
                    f"{model_name}/ood_acc_drop": acc_drop,
                }
            )

    return model, results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cora GCN/GAT Baseline – DRL-GNN Project Phase I"
    )
    parser.add_argument(
        "--model",
        default="gcn",
        choices=["gcn", "gat", "both"],
        help="Which model(s) to train (default: gcn)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Max training epochs (default: 200)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--patience", type=int, default=30, help="Early-stopping patience (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--ood", action="store_true", help="Run OOD perturbation evaluation"
    )
    parser.add_argument(
        "--ood-edge-drop",
        type=float,
        default=0.15,
        dest="ood_edge_drop",
        help="Fraction of edges to drop for OOD (default: 0.15)",
    )
    parser.add_argument(
        "--ood-feat-noise",
        type=float,
        default=0.1,
        dest="ood_feat_noise",
        help="Std of Gaussian feature noise for OOD (default: 0.1)",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Log metrics to Weights & Biases"
    )
    args = parser.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B init ──────────────────────────────────────────────────────────────
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Skipping W&B logging.")
    if use_wandb:
        wandb.init(
            project="drl-gnn-cora-baseline",
            config={
                "model": args.model,
                "epochs": args.epochs,
                "lr": args.lr,
                "patience": args.patience,
                "seed": args.seed,
                "ood": args.ood,
            },
            name=f"baseline_{args.model}_seed{args.seed}",
        )

    # ── Load Cora ─────────────────────────────────────────────────────────────
    print("\nLoading Cora (Planetoid)...")
    dataset = Planetoid(root="/tmp/Cora", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    print(f"  Nodes       : {data.num_nodes}")
    print(f"  Edges       : {data.num_edges}")
    print(f"  Node features: {data.num_node_features}")
    print(f"  Classes     : {dataset.num_classes}")
    print(
        f"  Train / Val / Test nodes: "
        f"{data.train_mask.sum().item()} / "
        f"{data.val_mask.sum().item()} / "
        f"{data.test_mask.sum().item()}"
    )

    # ── Run chosen model(s) ──────────────────────────────────────────────────
    models_to_run = ["gcn", "gat"] if args.model == "both" else [args.model]
    all_results = []

    for model_name in models_to_run:
        _, res = run_training(model_name, data, args, device, use_wandb)
        all_results.append(res)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  BASELINE SUMMARY")
    print(f"{'='*55}")
    header = f"  {'Model':<6}  {'Test Acc':>8}  {'Val Acc':>8}  "
    header += f"{'Time(s)':>8}  {'ms/epoch':>9}"
    if args.ood:
        header += f"  {'OOD Acc':>8}  {'Drop':>6}"
    print(header)
    print(f"  {'-'*50}")
    for r in all_results:
        row = (
            f"  {r['model'].upper():<6}  {r['test_acc']:>8.4f}  "
            f"{r['best_val_acc']:>8.4f}  "
            f"{r['total_time_s']:>8.1f}  "
            f"{r['avg_epoch_ms']:>9.1f}"
        )
        if args.ood:
            row += (
                f"  {r.get('ood_test_acc', float('nan')):>8.4f}  "
                f"{r.get('ood_acc_drop', float('nan')):>6.4f}"
            )
        print(row)
    print()
    print(
        "  Checkpoints saved: "
        + ", ".join(f"best_{r['model']}.pt" for r in all_results)
    )
    print("  Use these checkpoints as the reference when comparing Paradigm A/B.")

    if use_wandb:
        for r in all_results:
            wandb.summary[f"{r['model']}_test_acc"] = r["test_acc"]
            wandb.summary[f"{r['model']}_total_time"] = r["total_time_s"]
        wandb.finish()


if __name__ == "__main__":
    main()
