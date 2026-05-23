import numpy as np
from .multiverse import HierarchicalMultiverse
from .visualization import plot_evolution

def demo():
    print("=== Демонстрация иерархического любовного роя ===")
    multiverse = HierarchicalMultiverse(
        num_levels=3,
        agents_per_level=8,
        dim=8,
        lambda_=0.1,
        alpha=1.0,
        beta=1.0,
        lr=0.02
    )

    # История для визуализации
    history = {
        'foam': [],
        'S': [],
        'M_trajectories': []  # будем сохранять M всех агентов нижнего уровня
    }

    def callback(it, foam, levels):
        history['foam'].append(foam)
        # Субъектность всех агентов нижнего уровня
        lower_swarm = levels[0]
        history['S'].append([a.S for a in lower_swarm.agents])
        # Траектории M нижнего уровня
        history['M_trajectories'].append([a.M.copy() for a in lower_swarm.agents])

    multiverse.run(max_iters=300, tol=1e-8, callback=callback)

    # Визуализация
    plot_evolution(history, save_path="multiverse_evolution.png")
    print("График сохранён в multiverse_evolution.png")

if __name__ == "__main__":
    demo()
