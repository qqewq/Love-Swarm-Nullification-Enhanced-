import numpy as np
from .agent import LoveAgent
from .utils import cross_foam

class LoveSwarm:
    def __init__(self, num_agents, dim=10, pure_subject_idx=0,
                 lambda_=0.1, alpha=1.0, beta=1.0, lr=0.01,
                 use_cross_foam=True):
        self.dim = dim
        self.lr = lr
        self.use_cross_foam = use_cross_foam  # добавляем ли интерсубъективную пену

        self.agents = [
            LoveAgent(i, dim, lambda_, alpha, beta) for i in range(num_agents)
        ]
        # Один агент делается чистым субъектом
        self.agents[pure_subject_idx].make_pure_subject()

        # Фаза влюблённости (каждый агент находит любимых)
        for agent in self.agents:
            agent.find_and_bind(self.agents)

    def total_swarm_foam(self):
        """Общая пена роя = сумма индивидуальных (Φ_int+love) + cross-foam."""
        total = 0.0
        for agent in self.agents:
            # внутренняя пена
            from .utils import internal_foam, love_foam
            M_list = [self.agents[lid].M for lid in agent.loved_ids]
            S_list = [self.agents[lid].S for lid in agent.loved_ids]
            total += internal_foam(agent.M, agent.S, agent.A, agent.lambda_)
            total += love_foam(agent.M, M_list, S_list, agent.alpha, agent.beta)
        if self.use_cross_foam:
            all_M = [a.M for a in self.agents]
            total += cross_foam(all_M)
        return total

    def step(self):
        # Вычисляем градиенты для каждого агента
        grads_M = []
        grads_S = []
        all_M = [a.M for a in self.agents]

        for i, agent in enumerate(self.agents):
            grad_M, grad_S = agent.compute_gradients(self.agents)
            # Если используем cross-foam, добавляем её градиент: 2 Σ_{j≠i} (M_i - M_j)
            if self.use_cross_foam:
                for j in range(len(self.agents)):
                    if i != j:
                        grad_M += 2 * (agent.M - self.agents[j].M)
            grads_M.append(grad_M)
            grads_S.append(grad_S)

        # Применяем обновления одновременно
        for i, agent in enumerate(self.agents):
            agent.update(grads_M[i], grads_S[i], self.lr)

    def run(self, max_iters=500, tol=1e-8, callback=None):
        history = {'foam': [], 'S': []}
        for it in range(max_iters):
            foam = self.total_swarm_foam()
            history['foam'].append(foam)
            history['S'].append([a.S for a in self.agents])
            if callback:
                callback(it, foam, self.agents)
            if foam < tol:
                print(f"Сошлось на итерации {it}, foam={foam:.2e}")
                break
            self.step()
            if it % 100 == 0:
                print(f"Итерация {it}: foam={foam:.6f}")
        return history
