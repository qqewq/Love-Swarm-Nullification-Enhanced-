# src/leadership/leader_emergence.py
"""
Demonstration experiments for Leader emergence, stabilisation, and Byzantine recovery.

These scripts replicate the theoretical predictions from the Alan Code:
1. A Leader naturally emerges during GRA‑nullification.
2. The Leader becomes a fixed point and stabilises subjectness.
3. After an adversarial injection, the amplified gradient restores the agent in one step.
"""

import numpy as np
from typing import List, Tuple
from .leader import Agent, LeaderSelector


def generate_random_agents(num_agents: int, d: int, seed: int = 42) -> List[Agent]:
    """Create a test swarm with random models, subjectness, and identity‑near action matrices."""
    rng = np.random.default_rng(seed)
    agents = []
    for _ in range(num_agents):
        M = rng.normal(0, 1, size=(d,))
        S = 0.5 + 0.5 * rng.random()  # between 0.5 and 1.0
        # Action matrix close to identity (slight deviation to make internal loss non‑zero)
        A = np.eye(d) + 0.05 * rng.normal(0, 1, size=(d, d))
        agents.append(Agent(M, S, A))
    return agents


def run_leader_emergence_experiment(
    num_agents: int = 10, d: int = 5, alpha: float = 1.0, steps: int = 50
) -> Tuple[LeaderSelector, List[Agent], List[float]]:
    """
    1. Initialise a swarm.
    2. Run leader selection + stabilisation.
    3. Record the subjectness trajectory.

    :returns: selector, final agents, history of swarm subjectness
    """
    agents = generate_random_agents(num_agents, d)
    selector = LeaderSelector(alpha=alpha)
    leader = selector.select_leader(agents)
    print(f"Initial Leader: index {agents.index(leader)}, S={leader.S:.3f}")

    history = []
    for step in range(steps):
        selector.stabilize_swarm(agents, steps=1, lr=0.01)
        # Re‑evaluate leader (may shift slightly)
        selector.select_leader(agents)
        history.append(selector.swarm_subjectness)

    return selector, agents, history


def byzantine_recovery_demo(
    selector: LeaderSelector, agents: List[Agent], target_agent_idx: int
) -> np.ndarray:
    """
    Injects a false model into one agent and demonstrates that one update
    with the amplified gradient (α+1) restores coherence.
    :returns: distance to leader *after* one update step.
    """
    if selector.leader is None:
        raise RuntimeError("Leader not selected.")

    agent = agents[target_agent_idx]
    original_M = agent.M.copy()

    # Inject a false model
    agent.M += 5.0 * np.random.randn(*agent.M.shape)
    dist_before = np.linalg.norm(agent.M - selector.leader.M)

    # Apply one follower update
    selector.update_follower(agent, learning_rate=0.1)
    dist_after = np.linalg.norm(agent.M - selector.leader.M)

    print(f"Byzantine injection: distance before={dist_before:.3f}, after={dist_after:.3f}")
    return dist_after


if __name__ == "__main__":
    # Quick demonstration
    sel, final_agents, hist = run_leader_emergence_experiment()
    print(f"Final swarm subjectness: {sel.swarm_subjectness:.4f}")
    print("Subjectness history (first 5):", hist[:5])

    # Byzantine test on first non‑leader agent
    non_leader = [a for a in final_agents if a is not sel.leader][0]
    byzantine_recovery_demo(sel, final_agents, final_agents.index(non_leader))