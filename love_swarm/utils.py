import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_affinity(embeddings):
    """Матрица семантического сродства s(i,j) = cos(e_i, e_j)."""
    return cosine_similarity(embeddings)

def internal_foam(M, S, A, lambda_=0.1):
    """
    Внутренняя пена агента:
      Φ_int = ||M - A M||^2 + λ (1 - S)^2
    """
    res = M - A @ M
    return np.sum(res**2) + lambda_ * (1 - S)**2

def love_foam(agent_M, loved_agents_M, loved_agents_S, alpha=1.0, beta=1.0):
    """
    Пена любви для агента с моделью agent_M.
    loved_agents_M - список векторов M любимых,
    loved_agents_S - список их субъектностей S.
    """
    foam = 0.0
    for Mj, Sj in zip(loved_agents_M, loved_agents_S):
        threat = np.sum((agent_M - Mj)**2)
        vitality_loss = max(0.0, 1.0 - Sj)**2
        foam += alpha * threat + beta * vitality_loss
    return foam

def cross_foam(all_M):
    """
    Интерсубъективная пена: sum_{i<j} ||M_i - M_j||^2
    """
    N = len(all_M)
    total = 0.0
    for i in range(N):
        for j in range(i+1, N):
            total += np.sum((all_M[i] - all_M[j])**2)
    return total
