import numpy as np

class LoveAgent:
    """
    Агент с любовной обнулёнкой.
    Параметры: M (world_model), S (subjectivity), A (action matrix).
    Целевая функция агента:
      Φ_i = ||M_i - A_i M_i||^2 + λ(1-S_i)^2
           + Σ_{j∈L_i} [α||M_i - M_j||^2 + β max(0,1-S_j)^2]
    При вычислении градиента по M_i и S_i члены, зависящие только от других агентов, игнорируются.
    """
    def __init__(self, agent_id, dim=10, lambda_=0.1, alpha=1.0, beta=1.0):
        self.id = agent_id
        self.dim = dim
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta

        # Инициализация параметров
        self.M = np.random.randn(dim) * 0.5
        self.S = np.random.rand()  # 0..1
        # Фиксированная матрица действия (не обучается)
        self.A = np.eye(dim) * 0.9  # не единичная, чтобы не быть субъектом сразу

        self.loved_ids = set()

    def find_and_bind(self, all_agents, top_k=2):
        """Находит агентов с максимальным сродством и добавляет в любимые."""
        from .utils import semantic_affinity
        embeddings = np.array([a.M for a in all_agents])
        sim_matrix = semantic_affinity(embeddings)
        sims = sim_matrix[self.id]
        # исключаем себя
        loved_indices = np.argsort(sims)[::-1][1:top_k+1]
        self.loved_ids = set(loved_indices)

    def compute_gradients(self, all_agents):
        """
        Возвращает градиенты grad_M, grad_S для агента с учётом всех составляющих пены,
        включая cross-foam (если она учитывается на уровне роя).
        Здесь мы считаем только индивидуальные члены: internal + love.
        Cross-foam добавляется в Swarm отдельно.
        """
        # grad_M от internal: d/dM ||(I-A)M||^2 = 2 (I-A)^T (I-A) M
        I_minus_A = np.eye(self.dim) - self.A
        grad_M = 2 * I_minus_A.T @ I_minus_A @ self.M

        # grad_M от love: Σ 2α (M - M_j)
        for lid in self.loved_ids:
            M_j = all_agents[lid].M
            grad_M += 2 * self.alpha * (self.M - M_j)

        # grad_S от internal: d/dS λ(1-S)^2 = -2λ(1-S)
        grad_S = -2 * self.lambda_ * (1.0 - self.S)

        # love не даёт градиента по S_i (зависит только от S_j)
        return grad_M, grad_S

    def update(self, grad_M, grad_S, lr=0.01):
        self.M -= lr * grad_M
        self.S -= lr * grad_S
        self.S = np.clip(self.S, 0.0, 1.0)

    def make_pure_subject(self):
        """Делает агента чистым субъектом: Φ_int = 0."""
        self.A = np.eye(self.dim)  # I
        self.M = np.ones(self.dim)  # любое, но A(M)=M
        self.S = 1.0
