import numpy as np
from typing import List, Optional
from swarm import LoveSwarm

class Multiverse:
    """Иерархическая мультивселенная с проекциями Q_l и обратной связью γ."""

    def __init__(self, swarms: List[LoveSwarm], gamma: float = 0.1,
                 projection_basis: str = 'pca'):
        """
        swarms: список роев (уровней).
        gamma: сила обратной связи от высших уровней к низшим.
        projection_basis: 'pca' – базис главных компонент текущего консенсуса,
                          'fixed' – фиксированный случайный базис.
        """
        self.swarms = swarms
        self.gamma = gamma
        self.projection_basis = projection_basis
        self.levels = len(swarms)
        # Для PCA-проекций будем вычислять Q_l динамически
        self.Q = [None] * self.levels

    def compute_consensus(self, level: int) -> np.ndarray:
        """Консенсусная модель роя (среднее)."""
        agents = self.swarms[level].agents
        return np.mean([a.M for a in agents], axis=0)

    def compute_projection_matrix(self, level: int):
        """Вычисляет проекцию Q_l для уровня l."""
        if self.projection_basis == 'pca':
            # Базис из главных компонент моделей роя
            M_all = np.array([a.M for a in self.swarms[level].agents])
            # Центрируем
            M_centered = M_all - np.mean(M_all, axis=0)
            if M_centered.shape[0] > 1:
                cov = M_centered.T @ M_centered / (M_centered.shape[0] - 1)
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Базис: сортируем по убыванию собств. значений
                idx = np.argsort(eigvals)[::-1]
                self.Q[level] = eigvecs[:, idx]
            else:
                self.Q[level] = np.eye(self.swarms[level].dim)
        else:
            # Фиксированный случайный ортонормированный базис
            if self.Q[level] is None:
                Q = np.random.randn(self.swarms[level].dim, self.swarms[level].dim)
                self.Q[level] = np.linalg.qr(Q)[0]

    def upward_pass(self):
        """Снизу вверх: проецируем консенсус уровня l на уровень l+1."""
        for l in range(self.levels - 1):
            if self.Q[l] is None or self.projection_basis == 'pca':
                self.compute_projection_matrix(l)
            cons = self.compute_consensus(l)
            # Проекция в координатах Q_l
            proj = self.Q[l].T @ cons
            # Передаём как дополнительный сигнал агентам уровня l+1
            # (упрощённо: добавляем к их M с весом gamma)
            for agent in self.swarms[l+1].agents:
                # Переводим проекцию обратно в исходное пространство
                feedback = self.Q[l] @ proj
                agent.M += self.gamma * (feedback - agent.M)  # притягиваем

    def downward_pass(self):
        """Сверху вниз: корректируем низшие уровни проекцией высших."""
        for l in range(1, self.levels):
            if self.Q[l] is None:
                self.compute_projection_matrix(l)
            cons_high = self.compute_consensus(l)
            # Проекция высокоуровневого консенсуса вниз
            proj_high = self.Q[l-1].T @ cons_high
            feedback = self.Q[l-1] @ proj_high
            for agent in self.swarms[l-1].agents:
                agent.M += self.gamma * (feedback - agent.M)

    def step_all(self, lr: float = 0.01):
        """Один шаг эволюции всех роев с иерархическими связями."""
        for swarm in self.swarms:
            swarm.step(lr)
        self.upward_pass()
        self.downward_pass()