# examples/leader_integration.py
"""
Demonstrates integration of the Leader module into a swarm training loop.
Compares convergence speed with and without the Leader.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.leadership.leader import Agent, LeaderSelector

# --- Simulation Parameters ---
DIM = 4
NUM_AGENTS = 8
ALPHA = 1.0
STEPS = 40
LR = 0.01
RNG = np.random.default_rng(123)

def generate_agents():
    agents = []
    for _ in range(NUM_AGENTS):
        M = RNG.normal(0, 2, size=(DIM,))
        S = 0.4 + 0.4 * RNG.random()
        A = np.eye(DIM) + 0.03 * RNG.normal(0, 1, size=(DIM, DIM))
        agents.append(Agent(M, S, A))
    return agents

def standard_love_gradient(agent, others, alpha):
    """Standard Love Foam gradient (no leader)."""
    grad = np.zeros_like(agent.M)
    for other in others:
        if other is agent:
            continue
        grad += alpha * (agent.M - other.M)
    return 2.0 * grad  # from ∂/∂M_i of sum ||M_i-M_j||^2

def measure_coherence(agents):
    """Average pairwise distance between models."""
    dist = 0.0
    count = 0
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            dist += np.linalg.norm(agents[i].M - agents[j].M)
            count += 1
    return dist / count if count else 0.0

# ---- Baseline: standard swarm without Leader ----
agents_base = generate_agents()
print("=== Baseline (no Leader) ===")
for step in range(STEPS):
    old_agents = [Agent(a.M.copy(), a.S, a.A.copy()) for a in agents_base]
    for agent in agents_base:
        grad = standard_love_gradient(agent, agents_base, ALPHA)
        agent.M -= LR * grad
        agent.S += LR * 2.0 * (1.0 - agent.S)  # subjectness drive
    if step % 10 == 0:
        print(f"Step {step:2d}, coherence: {measure_coherence(agents_base):.4f}")

print(f"Final coherence without leader: {measure_coherence(agents_base):.6f}")

# ---- With Leader ----
agents_leader = generate_agents()
selector = LeaderSelector(alpha=ALPHA)
print("\n=== With Leader (activated at step 0) ===")
selector.select_leader(agents_leader)
for step in range(STEPS):
    selector.stabilize_swarm(agents_leader, steps=1, lr=LR)
    if step % 10 == 0:
        print(f"Step {step:2d}, coherence: {measure_coherence(agents_leader):.4f}")

print(f"Final coherence with leader: {measure_coherence(agents_leader):.6f}")
print(f"Swarm subjectness: {selector.swarm_subjectness:.4f}")