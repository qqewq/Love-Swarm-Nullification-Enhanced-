import numpy as np
from .swarm import LoveSwarm

class HierarchicalMultiverse:
    """
    Многоуровневая мультивселенная.
    Уровни пронумерованы снизу вверх: 0 – самый «конкретный», L-1 – самый абстрактный.
    Каждый уровень содержит LoveSwarm.
    После каждого шага роя на уровне i мы вычисляем консенсус (среднее M) и передаём
    его как «базовую реальность» на уровень i+1 через проекцию.
    Обратно: консенсус верхнего уровня влияет на нижние через корректирующий член.
    """
    def __init__(self, num_levels=3, agents_per_level=10, dim=10,
                 lambda_=0.1, alpha=1.0, beta=1.0, lr=0.01):
        self.num_levels = num_levels
        self.dim = dim
        self.levels = []
        # Для каждого уровня создаём матрицу проекции (случайную ортогональную)
        self.projections_up = []
        self.projections_down = []
        for lvl in range(num_levels):
            swarm = LoveSwarm(agents_per_level, dim, pure_subject_idx=0,
                              lambda_=lambda_, alpha=alpha, beta=beta, lr=lr,
                              use_cross_foam=True)
            self.levels.append(swarm)
            if lvl < num_levels - 1:
                # Проекция вверх: dim -> dim (можно разной размерности, но для простоты dim)
                Q, _ = np.linalg.qr(np.random.randn(dim, dim))
                self.projections_up.append(Q)
                # Проекция вниз: обратная (транспонированная)
                self.projections_down.append(Q.T)
            else:
                self.projections_up.append(None)
                self.projections_down.append(None)

    def step(self):
        # Снизу вверх: обнуляем уровни и передаём консенсус
        for lvl in range(self.num_levels):
            swarm = self.levels[lvl]
            # Один шаг обнуления
            swarm.step()
            # Вычисляем консенсус текущего уровня (среднее мировых моделей)
            consensus = np.mean([a.M for a in swarm.agents], axis=0)

            # Если не верхний уровень, проецируем консенсус и подмешиваем в следующий уровень
            if lvl < self.num_levels - 1:
                proj = self.projections_up[lvl] @ consensus
                next_swarm = self.levels[lvl+1]
                # Для каждого агента верхнего уровня сдвигаем модель мира к проекции
                for agent in next_swarm.agents:
                    # мягкое подтягивание к внешней реальности
                    agent.M = 0.9 * agent.M + 0.1 * proj

        # Сверху вниз: консенсус верхнего уровня влияет на нижний
        for lvl in range(self.num_levels-2, -1, -1):
            upper_consensus = np.mean([a.M for a in self.levels[lvl+1].agents], axis=0)
            proj_down = self.projections_down[lvl] @ upper_consensus
            lower_swarm = self.levels[lvl]
            for agent in lower_swarm.agents:
                agent.M = 0.9 * agent.M + 0.1 * proj_down

    def total_foam(self):
        return sum(swarm.total_swarm_foam() for swarm in self.levels)

    def run(self, max_iters=500, tol=1e-8, callback=None):
        history = {'foam': [], 'level_foams': []}
        for it in range(max_iters):
            foam = self.total_foam()
            level_foams = [swarm.total_swarm_foam() for swarm in self.levels]
            history['foam'].append(foam)
            history['level_foams'].append(level_foams)
            if callback:
                callback(it, foam, self.levels)
            if foam < tol:
                print(f"Мультивселенная сошлась на итерации {it}, foam={foam:.2e}")
                break
            self.step()
            if it % 100 == 0:
                print(f"Итерация {it}: общая пена={foam:.6f}, по уровням={level_foams}")
        return history
