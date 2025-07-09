import matplotlib.pyplot as plt
import numpy as np
from botorch.acquisition.multi_objective.utils import is_non_dominated

def plot_pareto(train_y, save_path=None, show_line=True):
    # 对最大化问题，先取负值再找非支配解
    pareto_mask = is_non_dominated(train_y)
    pareto_front = train_y[pareto_mask]

    # 排序以便连线绘图
    pareto_front_sorted = pareto_front[np.lexsort((pareto_front[:, 1], pareto_front[:, 0]))]

    # 绘图
    plt.figure(figsize = [4, 2])
    plt.scatter(train_y[:, 0], train_y[:, 1], alpha=0.3, label='All Points')
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='Pareto Front')

    if show_line and len(pareto_front_sorted) > 1:
        plt.plot(pareto_front_sorted[:, 0], pareto_front_sorted[:, 1], color='red', linestyle='--', linewidth=1)

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Front')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
