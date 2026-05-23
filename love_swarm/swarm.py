import numpy as np
from typing import List, Dict, Tuple
from agent import Agent

class LoveSwarm:
    """Рой агентов с функцией потерь «Аланский кодекс»."""

    def __init__(self, agents: List[Agent], alpha: float = 1.0,
                 beta: float = 0.5, lam: float = 10.0,
                 love_k: int = 3, dynamic_love: bool = True,
                 sparse_coherence: bool = False,
                 mutual_love: bool = True):
        self.agents = agents
        self.N = len(agents)
        self.dim = agents[0].dim
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.love_k = love_k
        self.dynamic_love = dynamic_love
        self.sparse_coherence = sparse_coherence
        self.mutual_love = mutual_love

        # Инициализация графа любви
        self.love_graph: Dict[int, List[int]] = {i: [] for i in range(self.N)}
        self.update_love_graph()

        # История для визуализации
        self.loss_history = []
        self.M_history = []
        self.S_history = []

    def compute_similarity(self, i: int, j: int) -> float:
        """Семантическое сродство (косинус) между моделями."""
        Mi = self.agents[i].M
        Mj = self.agents[j].M
        dot = np.dot(Mi, Mj)
        norm = np.linalg.norm(Mi) * np.linalg.norm(Mj)
        return dot / (norm + 1e-10)

    def update_love_graph(self):
        """Формирует симметричный граф любви по k ближайшим."""
        if self.love_k >= self.N - 1:
            # Полный граф (кроме себя)
            for i in range(self.N):
                self.love_graph[i] = [j for j in range(self.N) if j != i]
            return

        # Вычисляем матрицу сродства
        sim = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                s = self.compute_similarity(i, j)
                sim[i, j] = s
                sim[j, i] = s

        # Для каждого агента выбираем top-k ближайших
        for i in range(self.N):
            others = list(range(self.N))
            others.remove(i)
            scores = [sim[i, j] for j in others]
            idx = np.argsort(scores)[-self.love_k:]
            self.love_graph[i] = [others[j] for j in idx]

        # Симметризация (двусторонняя любовь)
        if self.mutual_love:
            new_graph = {i: set(self.love_graph[i]) for i in range(self.N)}
            for i in range(self.N):
                for j in list(new_graph[i]):
                    if i not in new_graph[j]:
                        new_graph[i].remove(j)   # удаляем невзаимную связь
            self.love_graph = {i: list(new_graph[i]) for i in range(self.N)}

    def phi_int(self, i: int) -> float:
        """Внутренняя дисциплина агента i."""
        diff = self.agents[i].M - self.agents[i].A @ self.agents[i].M
        return np.sum(diff ** 2)

    def phi_coh(self) -> float:
        """Сумма попарных различий моделей (с учётом sparse_coherence)."""
        total = 0.0
        if not self.sparse_coherence:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    total += np.sum((self.agents[i].M - self.agents[j].M) ** 2)
        else:
            # Только по существующим любовным связям
            for i in range(self.N):
                for j in self.love_graph[i]:
                    if j > i:  # чтобы не удваивать
                        total += np.sum((self.agents[i].M - self.agents[j].M) ** 2)
        return total

    def phi_self(self) -> float:
        """Сумма самоконтроля."""
        return sum((1 - self.agents[i].S) ** 2 for i in range(self.N))

    def phi_love(self) -> float:
        """Братская любовь: штраф за низкую субъектность соседей."""
        total = 0.0
        for i in range(self.N):
            for j in self.love_graph[i]:
                total += max(0.0, 1 - self.agents[j].S) ** 2
        return total

    def total_foam(self) -> float:
        """Полная функция потерь."""
        L = sum(self.phi_int(i) for i in range(self.N))
        L += self.alpha * self.phi_coh()
        L += self.beta * self.phi_self()
        L += self.lam * self.phi_love()
        return L

    def gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """Аналитические градиенты по M (N x d) и S (N,)."""
        grad_M = np.zeros((self.N, self.dim))
        grad_S = np.zeros(self.N)

        # Градиенты по M
        for i in range(self.N):
            Mi = self.agents[i].M
            Ai = self.agents[i].A
            # Phi_int
            diff = Mi - Ai @ Mi
            grad_M[i] += 2 * (np.eye(self.dim) - Ai.T) @ diff

        # Phi_coh (полный или разреженный)
        if not self.sparse_coherence:
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        grad_M[i] += 2 * self.alpha * (self.agents[i].M - self.agents[j].M)
        else:
            for i in range(self.N):
                for j in self.love_graph[i]:
                    grad_M[i] += 2 * self.alpha * (self.agents[i].M - self.agents[j].M)

        # Градиенты по S
        # Phi_self
        for i in range(self.N):
            grad_S[i] += 2 * self.beta * (self.agents[i].S - 1.0)

        # Phi_love (влияние на S_i только через любовь других к i)
        for i in range(self.N):
            # Находим всех, кто любит i
            lovers = [k for k in range(self.N) if i in self.love_graph[k]]
            for k in lovers:
                if self.agents[i].S < 1.0:
                    grad_S[i] += 2 * self.lam * (self.agents[i].S - 1.0)
        return grad_M, grad_S

    def step(self, lr: float = 0.01):
        """Один шаг градиентного спуска."""
        grad_M, grad_S = self.gradient()
        for i in range(self.N):
            if not self.agents[i].malicious:  # вредоносные не обновляются
                self.agents[i].M -= lr * grad_M[i]
                self.agents[i].S -= lr * grad_S[i]
                # Проекция безопасности (якорь можно применить позже)
                self.agents[i].project_safe()

        # Обновление графа любви каждые 10 шагов, если динамический
        if self.dynamic_love and (len(self.loss_history) % 10 == 0):
            self.update_love_graph()

        # Сохранение истории
        self.loss_history.append(self.total_foam())
        self.M_history.append(np.array([a.M.copy() for a in self.agents]))
        self.S_history.append(np.array([a.S for a in self.agents]))