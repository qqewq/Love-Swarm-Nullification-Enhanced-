import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from sklearn.decomposition import PCA   # требуется scikit-learn

def plot_convergence(swarm, title="Swarm Convergence"):
    """Три графика: лосс, среднее pairwise расстояние M, субъектности."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    # Лосс
    axs[0].plot(swarm.loss_history)
    axs[0].set_title('Total Foam')
    axs[0].set_xlabel('Step')
    # Среднее расстояние между M
    M_hist = np.array(swarm.M_history)
    N, steps, d = M_hist.shape[0], M_hist.shape[1], M_hist.shape[2]
    pairwise_dist = np.zeros(steps)
    for t in range(steps):
        dist_sum = 0.0
        cnt = 0
        for i in range(N):
            for j in range(i+1, N):
                dist_sum += np.linalg.norm(M_hist[i, t] - M_hist[j, t])
                cnt += 1
        pairwise_dist[t] = dist_sum / cnt
    axs[1].plot(pairwise_dist)
    axs[1].set_title('Mean pairwise M distance')
    # Субъектности
    S_hist = np.array(swarm.S_history).T  # (N, steps)
    for i in range(S_hist.shape[0]):
        axs[2].plot(S_hist[i], label=f'Agent {i}')
    axs[2].set_title('Subjectivity S')
    axs[2].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_pca_trajectory(swarm, safe_model: Optional[np.ndarray] = None):
    """PCA-траектория моделей (первые 2 компоненты)."""
    M_all = []
    for t in range(len(swarm.M_history)):
        M_all.append(swarm.M_history[t].reshape(-1, swarm.dim))
    M_all = np.vstack(M_all)
    pca = PCA(n_components=2)
    pca.fit(M_all)
    plt.figure(figsize=(8, 6))
    for i in range(swarm.N):
        traj = np.array([swarm.M_history[t][i] for t in range(len(swarm.M_history))])
        traj_2d = pca.transform(traj)
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], label=f'Agent {i}')
        plt.scatter(traj_2d[-1, 0], traj_2d[-1, 1])
    if safe_model is not None:
        safe_2d = pca.transform(safe_model.reshape(1, -1))
        plt.scatter(safe_2d[0, 0], safe_2d[0, 1], marker='*', s=200,
                    c='red', label='Safe Anchor')
    plt.title('PCA trajectory of M')
    plt.legend()
    plt.show()

def plot_affinity_matrix(swarm):
    """Тепловая карта взаимного сродства агентов."""
    N = swarm.N
    sim = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim[i, j] = swarm.compute_similarity(i, j)
    plt.imshow(sim, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Agent affinity matrix')
    plt.xlabel('Agent j')
    plt.ylabel('Agent i')
    plt.show()