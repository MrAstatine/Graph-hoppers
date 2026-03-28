"""
train_paradigmA.py
------------------
Phase III: Train the Paradigm A agent on Cora.

Pipeline:
  1. Load pre-trained GCN baseline (encoder + classifier head).
  2. Build CoraNodeClassEnv with the frozen encoder.
  3. Train SB3 PPO policy that acts on GCN-derived observations.
  4. Log metrics to W&B (optional) and CSV.

Usage:
    python train_paradigmA.py [--timesteps 100000] [--seed 42] [--wandb]

Typical Colab command:
    !python train_paradigmA.py --timesteps 50000 --seed 42

Outputs
-------
  - models/paradigmA_ppo_seed{seed}/   (SB3 checkpoint)
  - logs/paradigmA_ppo_seed{seed}.csv
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from gcn_encoder import GCNClassifier
from cora_env import CoraNodeClassEnv

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

DATA_ROOT = "./data"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Paradigm A PPO training on Cora")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget", type=int, default=20,
                   help="Nodes queried per episode")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--gcn_ckpt", type=str, default="",
                   help="Path to pre-trained GCN checkpoint (leave empty to skip)")
    p.add_argument("--perturb_ratio", type=float, default=0.0,
                   help="Set > 0 to train in OOD mode (usually keep at 0)")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# W&B callback (no-op if W&B not requested)
# ---------------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    """
    Logs episode reward mean to CSV (and W&B if enabled).
    """

    def __init__(self, log_path: str, use_wandb: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.use_wandb = use_wandb
        self._csv = open(log_path, "w", newline="")
        self._writer = csv.writer(self._csv)
        self._writer.writerow(["timestep", "ep_rew_mean", "ep_len_mean", "wall_time_s"])
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            ep_len = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
            elapsed = time.time() - self._t0
            self._writer.writerow([
                self.num_timesteps,
                f"{ep_rew:.4f}",
                f"{ep_len:.1f}",
                f"{elapsed:.1f}",
            ])
            self._csv.flush()

            if self.use_wandb:
                import wandb
                wandb.log({
                    "ep_rew_mean": ep_rew,
                    "ep_len_mean": ep_len,
                    "timestep": self.num_timesteps,
                })
        return True

    def _on_training_end(self):
        self._csv.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random; random.seed(seed)


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Build GCN model
    # ------------------------------------------------------------------
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    dataset = Planetoid(root=DATA_ROOT, name="Cora", transform=NormalizeFeatures())
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    model = GCNClassifier(
        in_channels=num_features,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    # Load pre-trained weights if provided
    if args.gcn_ckpt and os.path.isfile(args.gcn_ckpt):
        model.load_state_dict(torch.load(args.gcn_ckpt, map_location=device))
        print(f"[INFO] Loaded GCN checkpoint: {args.gcn_ckpt}")
    else:
        print("[INFO] No GCN checkpoint specified; encoder starts from scratch.")
        print("       Tip: run train_baseline.py first for a warm encoder.")

    # Freeze encoder so the RL observation space stays stable
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()

    # ------------------------------------------------------------------
    # 2. Build environment
    # ------------------------------------------------------------------
    env = CoraNodeClassEnv(
        encoder=model.encoder,
        split="test",
        budget=args.budget,
        perturb_ratio=args.perturb_ratio,
        seed=args.seed,
        device=str(device),
    )
    # Inject the trained classifier head so reward is meaningful
    env.set_classifier(model.classifier)
    env = Monitor(env)

    # ------------------------------------------------------------------
    # 3. Optional W&B init
    # ------------------------------------------------------------------
    if args.wandb:
        import wandb
        wandb.init(
            project="paradigmA-cora",
            name=f"ppo_seed{args.seed}",
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # 4. Build PPO agent
    # ------------------------------------------------------------------
    save_dir = f"models/paradigmA_ppo_seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        seed=args.seed,
        tensorboard_log=None,
    )

    log_path = f"logs/paradigmA_ppo_seed{args.seed}.csv"
    callback = MetricsCallback(log_path=log_path, use_wandb=args.wandb)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(f"\nTraining PPO (Paradigm A) for {args.timesteps} timesteps, seed={args.seed}")
    t_start = time.time()
    agent.learn(total_timesteps=args.timesteps, callback=callback)
    train_time = time.time() - t_start

    agent.save(os.path.join(save_dir, "ppo_final"))
    print(f"\n[INFO] Saved PPO agent to {save_dir}/ppo_final.zip")
    print(f"[INFO] Total training time: {train_time:.1f}s")
    print(f"[INFO] Metrics log: {log_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
