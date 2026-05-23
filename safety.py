import numpy as np
from typing import List
from agent import Agent

class SafetyAnchor:
    """Эталонная безопасная модель и порог допустимого отклонения."""
    def __init__(self, safe_model: np.ndarray, threshold: float = 2.0):
        self.safe_model = safe_model / np.linalg.norm(safe_model)
        self.threshold = threshold

    def check(self, M: np.ndarray) -> bool:
        dist = np.linalg.norm(M - self.safe_model)
        return dist <= self.threshold

    def project(self, M: np.ndarray) -> np.ndarray:
        diff = M - self.safe_model
        dist = np.linalg.norm(diff)
        if dist > self.threshold:
            diff = diff / dist * self.threshold
            return self.safe_model + diff
        return M

class AuditorNode:
    """Аудитор: проверяет агентов, временно или навсегда исключает нарушителей."""
    def __init__(self, anchor: SafetyAnchor, quarantine_steps: int = 50,
                 permanent_ban_threshold: int = 3):
        self.anchor = anchor
        self.quarantine_steps = quarantine_steps
        self.permanent_ban_threshold = permanent_ban_threshold
        self.violation_counts = {}
        self.quarantine = set()
        self.banned = set()

    def audit(self, agents: List[Agent], love_graph: dict) -> dict:
        new_love = {i: list(love_graph[i]) for i in range(len(agents))}
        for i, agent in enumerate(agents):
            if i in self.banned:
                for j in range(len(agents)):
                    if i in new_love.get(j, []):
                        new_love[j].remove(i)
                new_love[i] = []
                continue

            if not self.anchor.check(agent.M):
                self.violation_counts[i] = self.violation_counts.get(i, 0) + 1
                if self.violation_counts[i] >= self.permanent_ban_threshold:
                    self.banned.add(i)
                else:
                    self.quarantine.add(i)
            else:
                if i in self.violation_counts:
                    del self.violation_counts[i]
                if i in self.quarantine:
                    self.quarantine.remove(i)

        for i in self.quarantine.union(self.banned):
            for j in range(len(agents)):
                if i in new_love.get(j, []):
                    new_love[j].remove(i)
            new_love[i] = []
        return new_love