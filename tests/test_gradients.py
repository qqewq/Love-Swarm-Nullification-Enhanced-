import numpy as np
import sys
sys.path.append('..')
from agent import Agent
from swarm import LoveSwarm

def test_gradient():
    np.random.seed(42)
    dim = 3
    N = 4
    agents = [Agent(dim) for _ in range(N)]
    swarm = LoveSwarm(agents, love_k=2)
    eps = 1e-5

    grad_M_num = np.zeros((N, dim))
    grad_S_num = np.zeros(N)

    M_orig = [a.M.copy() for a in agents]
    S_orig = [a.S for a in agents]

    for i in range(N):
        for d in range(dim):
            orig = M_orig[i][d]
            agents[i].M[d] = orig + eps
            Lp = swarm.total_foam()
            agents[i].M[d] = orig - eps
            Lm = swarm.total_foam()
            agents[i].M[d] = orig
            grad_M_num[i, d] = (Lp - Lm) / (2 * eps)

    for i in range(N):
        orig = S_orig[i]
        agents[i].S = orig + eps
        Lp = swarm.total_foam()
        agents[i].S = orig - eps
        Lm = swarm.total_foam()
        agents[i].S = orig
        grad_S_num[i] = (Lp - Lm) / (2 * eps)

    grad_M_anal, grad_S_anal = swarm.gradient()
    max_diff_M = np.abs(grad_M_anal - grad_M_num).max()
    max_diff_S = np.abs(grad_S_anal - grad_S_num).max()
    print(f"Max gradient diff M: {max_diff_M:.2e}, S: {max_diff_S:.2e}")
    assert max_diff_M < 1e-6, "Gradient M test failed"
    assert max_diff_S < 1e-6, "Gradient S test failed"
    print("All gradient tests passed!")

if __name__ == "__main__":
    test_gradient()