# src/leadership/leader.py
"""
Formal Leader selection and swarm stabilisation for agent swarms under the Alan Code.

Mathematical foundation:
- Leader := argmax_i [ S_i * (1 - ||M_i - M_cons|| / max_j ||M_j - M_cons||) ]
- Follower gradient: ∇_{M_i} L = 2(I - A_i)ᵀ(I - A_i)M_i + 2(α+1)(M_i - M_Leader)
- Swarm subjectness is set to S_Leader, which serves as an invariant norm.
"""

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class Agent:
    """
    Minimal agent representation sufficient for the leadership module.
    In a full integration these fields would be provided by the swarm's Agent class.
    """
    M: np.ndarray          # world model, shape (d,)
    S: float               # subjectness ∈ [0,1]
    A: np.ndarray          # action matrix, shape (d,d)


class LeaderSelector:
    """
    Encapsulates the logic for choosing and applying the swarm Leader.
    """

    def __init__(self, alpha: float = 1.0):
        """
        :param alpha: love‑foam coupling strength (used in amplified gradient).
        """
        self.alpha = alpha
        self.leader: Agent = None
        self.swarm_subjectness: float = 0.0

    def select_leader(self, agents: List[Agent]) -> Agent:
        """
        Identifies the Leader from a list of agents using the gravitational‑mass criterion.

        Leader = argmax_i [ S_i * (1 - ||M_i - M_cons|| / max_j ||M_j - M_cons||) ]

        :param agents: list of Agent objects (all must have non‑empty M)
        :return: the chosen Leader agent
        """
        # Compute consensus model as the simple mean (in GRA‑nullified state, models are aligned;
        # for emergence we use the mean of the current models).
        M_stack = np.stack([a.M for a in agents], axis=0)
        M_cons = np.mean(M_stack, axis=0)

        # Distance of each agent from consensus
        distances = np.array([np.linalg.norm(a.M - M_cons) for a in agents])
        max_dist = distances.max() if distances.max() > 1e-12 else 1.0

        # Leader score
        scores = np.array([a.S * (1.0 - distances[i] / max_dist) for i, a in enumerate(agents)])

        leader_idx = np.argmax(scores)
        self.leader = agents[leader_idx]
        self.swarm_subjectness = self.leader.S
        return self.leader

    def follower_gradient(self, agent: Agent) -> np.ndarray:
        """
        Computes the gradient of the Alan Loss w.r.t. the follower's model M_i,
        assuming the Leader is fixed.

        ∇_{M_i} L = 2(I - A_i)ᵀ(I - A_i) M_i + 2(α+1)(M_i - M_Leader)

        :param agent: follower agent
        :return: gradient vector of same shape as agent.M
        """
        if self.leader is None:
            raise ValueError("Leader not selected. Call select_leader() first.")

        I = np.eye(agent.M.shape[0])
        diff = I - agent.A
        internal_term = 2.0 * diff.T @ diff @ agent.M

        love_term = 2.0 * (self.alpha + 1.0) * (agent.M - self.leader.M)
        return internal_term + love_term

    def update_follower(self, agent: Agent, learning_rate: float = 0.01) -> None:
        """
        Performs one gradient‑step for a follower, pulling its model toward the Leader.
        Subjectness is also nudged upward (as dictated by ∂L/∂S).
        """
        grad = self.follower_gradient(agent)
        agent.M -= learning_rate * grad

        # Push subjectness toward 1 (the Leader's S is the norm)
        agent.S += learning_rate * 2.0 * (1.0 - agent.S)

    def stabilize_swarm(self, agents: List[Agent], steps: int = 100, lr: float = 0.01) -> None:
        """
        Iteratively applies the Leader‑centred updates to all non‑Leader agents,
        driving the swarm to full coherence.
        """
        for _ in range(steps):
            for agent in agents:
                if agent is self.leader:
                    continue
                self.update_follower(agent, lr)