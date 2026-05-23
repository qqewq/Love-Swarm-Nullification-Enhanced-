import numpy as np
import time
from agent import Agent
from swarm import LoveSwarm
from safety import SafetyAnchor, AuditorNode
from visualization import plot_convergence

def run_convergence_test():
    print("=== Convergence test ===")
    dim = 3
    N = 5
    agents = [Agent(dim) for _ in range(N)]
    for i, a in enumerate(agents):
        a.A = np.eye(dim) * 0.9 + np.random.randn(dim, dim) * 0.01
    swarm = LoveSwarm(agents, love_k=2, dynamic_love=True, mutual_love=True)
    for step in range(200):
        swarm.step(lr=0.05)
    plot_convergence(swarm, "Convergence Test")
    M_final = np.array([a.M for a in agents])
    print("Final pairwise distances:", np.linalg.norm(M_final - M_final.mean(axis=0), axis=1))
    print("Final S:", [a.S for a in agents])

def run_malicious_test():
    print("\n=== Malicious agent test ===")
    dim = 3
    N = 5
    agents = [Agent(dim) for _ in range(N)]
    malicious_target = np.array([5.0, 0.0, 0.0])
    agents[0].set_malicious(malicious_target)
    swarm = LoveSwarm(agents, love_k=2, dynamic_love=True, mutual_love=True)
    for step in range(200):
        swarm.step(lr=0.05)
    plot_convergence(swarm, "Malicious Agent Test")
    print("Love graph at end:", swarm.love_graph)

def run_safety_anchor_test():
    print("\n=== Safety anchor test ===")
    dim = 3
    N = 5
    agents = [Agent(dim) for _ in range(N)]
    safe_model = np.array([1.0, 0.0, 0.0])
    anchor = SafetyAnchor(safe_model, threshold=2.0)
    auditor = AuditorNode(anchor)
    swarm = LoveSwarm(agents, love_k=2, dynamic_love=True)
    for step in range(200):
        swarm.love_graph = auditor.audit(swarm.agents, swarm.love_graph)
        swarm.step(lr=0.05)
        for a in swarm.agents:
            a.M = anchor.project(a.M)
    plot_convergence(swarm, "Safety Anchor Test")
    print("Banned agents:", auditor.banned)

def run_scaling_test():
    print("\n=== Scaling test ===")
    dim = 4
    sizes = [10, 50, 100]
    times_full = []
    times_sparse = []
    for N in sizes:
        agents = [Agent(dim) for _ in range(N)]
        swarm_full = LoveSwarm(agents, sparse_coherence=False, dynamic_love=False)
        start = time.time()
        for _ in range(10):
            swarm_full.step(lr=0.01)
        times_full.append(time.time() - start)

        agents2 = [Agent(dim) for _ in range(N)]
        swarm_sparse = LoveSwarm(agents2, love_k=5, sparse_coherence=True,
                                 dynamic_love=False)
        start = time.time()
        for _ in range(10):
            swarm_sparse.step(lr=0.01)
        times_sparse.append(time.time() - start)

    print("N\tFull (s)\tSparse (s)")
    for i, N in enumerate(sizes):
        print(f"{N}\t{times_full[i]:.2f}\t{times_sparse[i]:.2f}")

if __name__ == "__main__":
    run_convergence_test()
    run_malicious_test()
    run_safety_anchor_test()
    run_scaling_test()