"""
cora_env.py
-----------
Gymnasium environment for Paradigm A: GNN -> DRL on Cora.

Task: The DRL agent decides, at each step, which unlabelled node to
classify (i.e., which node's neighbourhood to "expand and predict").
The GCN encoder provides a condensed graph-level embedding + target-
node embedding as the observation. The reward is the accuracy signal
on the chosen node.

Episode structure
-----------------
- At reset, a random subset of `budget` unlabelled (val/test) nodes is
  chosen as the "query pool" for this episode.
- At each step the agent picks a node index (action) from the pool.
- The GCN runs on the CURRENT graph state (which may be perturbed for
  OOD evaluation) and predicts the label of the chosen node.
- Reward  = +1 if prediction correct, -1 otherwise.
  Optionally a small entropy-penalty term is added for confident wrong
  predictions.
- Episode ends when all budget nodes have been queried.

Observation space
-----------------
  [graph_emb (embed_dim) || target_node_emb (embed_dim)]
  shape: (2 * embed_dim,)

Action space
------------
  Discrete(budget)  - index into the current query pool.

OOD perturbation
----------------
  Controlled by `perturb_ratio` (fraction of edges/nodes to perturb).
  Call env.set_perturb(ratio) before reset() to enable OOD mode.
"""

import random
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_random_edge, dropout_edge


DATA_ROOT = "./data"
EMBED_DIM = 32   # must match GCNEncoder embed_dim


class CoraNodeClassEnv(gym.Env):
    """
    Gymnasium env wrapping the Cora node-classification task for
    Paradigm A (GNN-guided DRL).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        encoder,                   # pre-built GCNEncoder (or GCNClassifier.encoder)
        split: str = "test",       # which split to query: "val" or "test"
        budget: int = 20,          # nodes queried per episode
        perturb_ratio: float = 0.0,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__()

        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.split = split
        self.budget = budget
        self.perturb_ratio = perturb_ratio
        self.seed_val = seed
        self.device = device

        # ------------------------------------------------------------------
        # Load Cora
        # ------------------------------------------------------------------
        dataset = Planetoid(
            root=DATA_ROOT,
            name="Cora",
            transform=NormalizeFeatures(),
        )
        self.data = dataset[0].to(device)
        self.num_classes = dataset.num_classes

        # Build pool of candidate nodes (val or test mask)
        mask = self.data.val_mask if split == "val" else self.data.test_mask
        self.candidate_pool = mask.nonzero(as_tuple=True)[0].tolist()

        # ------------------------------------------------------------------
        # Spaces
        # ------------------------------------------------------------------
        obs_dim = 2 * EMBED_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(budget)

        # Internal state
        self._query_pool: list = []
        self._step_count: int = 0
        self._current_edge_index = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_perturb(self, ratio: float):
        """Enable OOD mode with given perturbation ratio (0.0 = clean)."""
        self.perturb_ratio = ratio

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        rng_seed = seed if seed is not None else self.seed_val
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)

        # Sample query pool for this episode
        pool_size = min(self.budget, len(self.candidate_pool))
        self._query_pool = random.sample(self.candidate_pool, pool_size)
        self._step_count = 0

        # Optionally perturb the graph
        self._current_edge_index = self._perturb_graph(self.data.edge_index)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert 0 <= action < len(self._query_pool), (
            f"Action {action} out of range [0, {len(self._query_pool)})"
        )

        target_node = self._query_pool[action]
        reward = self._compute_reward(target_node)

        self._step_count += 1
        terminated = self._step_count >= len(self._query_pool)

        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        info = {
            "target_node": target_node,
            "step": self._step_count,
        }
        return obs, reward, terminated, False, info

    def render(self):
        print(f"Step {self._step_count}/{len(self._query_pool)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perturb_graph(self, edge_index):
        """Apply random edge drops/additions for OOD testing."""
        if self.perturb_ratio <= 0.0:
            return edge_index
        ei, _ = dropout_edge(edge_index, p=self.perturb_ratio, force_undirected=True)
        ei, _ = add_random_edge(
            ei,
            p=self.perturb_ratio,
            num_nodes=self.data.num_nodes,
            force_undirected=True,
        )
        return ei

    @torch.no_grad()
    def _get_node_embeddings(self):
        """Run GCN encoder on current (possibly perturbed) graph."""
        return self.encoder(self.data.x, self._current_edge_index)  # [N, embed_dim]

    def _get_obs(self):
        """Concatenate graph-mean embedding with next target-node embedding."""
        node_emb = self._get_node_embeddings()  # [N, embed_dim]
        graph_emb = node_emb.mean(dim=0)         # [embed_dim]

        # Use the first node in remaining pool as "next target" hint
        next_target = self._query_pool[self._step_count]
        target_emb = node_emb[next_target]       # [embed_dim]

        obs = torch.cat([graph_emb, target_emb], dim=0).cpu().numpy()
        return obs.astype(np.float32)

    @torch.no_grad()
    def _compute_reward(self, target_node: int) -> float:
        """
        Reward = +1 if GCN predicts correct label for target_node, else -1.
        A small entropy bonus encourages confident correct predictions.
        """
        node_emb = self._get_node_embeddings()  # [N, embed_dim]
        # We need a classifier head; use the encoder's parent if available
        # or fall back to a simple dot-product with class prototypes
        # (trainer will inject the full classifier via set_classifier)
        if hasattr(self, "_classifier_head") and self._classifier_head is not None:
            logits = self._classifier_head(node_emb[target_node].unsqueeze(0))  # [1, C]
        else:
            # Fallback: no classifier yet, reward = 0
            return 0.0

        probs = F.softmax(logits, dim=-1).squeeze()  # [C]
        pred = probs.argmax().item()
        true_label = self.data.y[target_node].item()

        correct = float(pred == true_label)
        entropy_bonus = -float((probs * (probs + 1e-8).log()).sum()) / np.log(self.num_classes)

        # reward: +1 correct, -1 wrong; entropy bonus in [0,1] scaled small
        reward = (2 * correct - 1) + 0.1 * entropy_bonus
        return float(reward)

    def set_classifier(self, classifier_head):
        """Inject trained linear head so reward can be computed."""
        self._classifier_head = classifier_head
