# tests/test_leader.py
"""
Unit tests for the Leader module.
Validates leader emergence, fixed-point property, and Byzantine recovery.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Make sure we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.leadership.leader import Agent, LeaderSelector


class TestLeader(unittest.TestCase):
    def setUp(self):
        """Create a small deterministic swarm for reproducible tests."""
        self.d = 3
        self.num_agents = 5
        self.alpha = 1.0
        self.learning_rate = 0.1
        rng = np.random.default_rng(42)
        self.agents = []
        for _ in range(self.num_agents):
            M = rng.normal(0, 1, size=(self.d,))
            S = 0.5 + 0.5 * rng.random()
            A = np.eye(self.d) + 0.05 * rng.normal(0, 1, size=(self.d, self.d))
            self.agents.append(Agent(M, S, A))
        self.selector = LeaderSelector(alpha=self.alpha)

    def test_leader_emergence(self):
        """Leader should be the agent with highest gravitational mass score."""
        self.selector.select_leader(self.agents)
        leader = self.selector.leader
        # Compute consensus and scores manually
        M_stack = np.stack([a.M for a in self.agents])
        M_cons = np.mean(M_stack, axis=0)
        distances = np.array([np.linalg.norm(a.M - M_cons) for a in self.agents])
        max_dist = distances.max() if distances.max() > 1e-12 else 1.0
        scores = np.array([a.S * (1.0 - distances[i] / max_dist) for i, a in enumerate(self.agents)])
        expected_leader_idx = np.argmax(scores)
        actual_leader_idx = self.agents.index(leader)
        self.assertEqual(actual_leader_idx, expected_leader_idx,
                         "Leader selection must follow the gravitational-mass criterion.")

    def test_leader_fixed_point(self):
        """After stabilisation, the leader's gradient should be close to zero."""
        self.selector.select_leader(self.agents)
        self.selector.stabilize_swarm(self.agents, steps=50, lr=self.learning_rate)
        grad_leader = self.selector.follower_gradient(self.selector.leader)
        # Gradient for leader itself should be tiny (theoretically zero)
        self.assertTrue(np.linalg.norm(grad_leader) < 1e-6,
                        f"Leader gradient norm {np.linalg.norm(grad_leader)} should be near zero.")

    def test_byzantine_recovery(self):
        """An injected false model is corrected in a single follower update."""
        self.selector.select_leader(self.agents)
        # Stabilise first to have a coherent swarm
        self.selector.stabilize_swarm(self.agents, steps=30, lr=0.05)
        # Pick a non-leader agent
        follower = [a for a in self.agents if a is not self.selector.leader][0]
        M_leader = self.selector.leader.M.copy()
        # Corrupt follower's model
        corruption = 3.0 * np.ones(self.d)  # large displacement
        follower.M += corruption
        dist_before = np.linalg.norm(follower.M - M_leader)
        self.assertGreater(dist_before, 1.0, "Corruption must significantly separate the agent.")
        # Apply one update
        self.selector.update_follower(follower, learning_rate=self.learning_rate)
        dist_after = np.linalg.norm(follower.M - M_leader)
        # After one update, the distance should shrink dramatically
        self.assertLess(dist_after, dist_before / 2.0,
                        f"Byzantine recovery must reduce distance significantly, got {dist_after} vs {dist_before}")


if __name__ == "__main__":
    unittest.main()