import numpy as np
from typing import Optional

class Agent:
    """Агент с моделью мира M, субъектностью S и матрицей действия A."""

    def __init__(self, dim: int, A: Optional[np.ndarray] = None,
                 malicious_target: Optional[np.ndarray] = None,
                 max_norm: float = 10.0):
        self.dim = dim
        # Модель мира: случайный вектор единичной длины
        self.M = np.random.randn(dim)
        self.M /= np.linalg.norm(self.M)
        # Субъектность: начинается низкой
        self.S = np.random.rand() * 0.3 + 0.1
        # Матрица действия (по умолчанию - ослабленная единичная)
        if A is None:
            self.A = np.eye(dim) * 0.9
        else:
            self.A = A
        # Параметры безопасности
        self.max_norm = max_norm
        # Вредоносный режим
        self.malicious = False
        if malicious_target is not None:
            self.set_malicious(malicious_target)

    def set_malicious(self, target: np.ndarray):
        """Фиксирует модель мира в опасном направлении."""
        self.M = target / np.linalg.norm(target)
        self.malicious = True

    def is_safe(self) -> bool:
        """Проверка простого ограничения: норма M не превышает порога."""
        return np.linalg.norm(self.M) <= self.max_norm

    def project_safe(self):
        """Проекция модели в допустимую область."""
        norm = np.linalg.norm(self.M)
        if norm > self.max_norm:
            self.M *= self.max_norm / norm