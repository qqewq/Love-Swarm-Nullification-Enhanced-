import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_evolution(history, agents_history=None, save_path=None):
    """
    Строит графики:
    1. Общая пена роя (или мультивселенной) по итерациям.
    2. Субъектность S каждого агента (если передан agents_history — список списков S).
    3. Траектории мировых моделей в 2D PCA (если есть agents_history_M).
    """
    if history is None:
        return

    num_plots = 2
    if agents_history and len(agents_history) > 0 and isinstance(agents_history[0], list):
        num_plots = 3

    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))

    # 1. Пена
    ax = axes[0]
    ax.plot(history['foam'])
    ax.set_yscale('log')
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Общая пена')
    ax.set_title('Сходимость обнуления')
    ax.grid(True)

    # 2. Субъектность
    if num_plots >= 2:
        ax = axes[1]
        if 'S' in history:
            S_array = np.array(history['S'])  # итерации x агенты
            for i in range(S_array.shape[1]):
                ax.plot(S_array[:, i], alpha=0.7)
            ax.set_xlabel('Итерация')
            ax.set_ylabel('Субъектность S')
            ax.set_title('Эволюция субъектности')
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True)

    # 3. Траектории M (PCA)
    if num_plots == 3 and 'M_trajectories' in history:
        ax = axes[2]
        M_traj = np.array(history['M_trajectories'])  # (iters, agents, dim)
        n_iters, n_agents, dim = M_traj.shape
        # Применяем PCA ко всем точкам сразу
        all_points = M_traj.reshape(-1, dim)
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(all_points).reshape(n_iters, n_agents, 2)
        for i in range(n_agents):
            ax.plot(points_2d[:, i, 0], points_2d[:, i, 1], alpha=0.6)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title('Траектории мировых моделей (PCA)')
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"График сохранён в {save_path}")
    else:
        plt.show()
    plt.close()
