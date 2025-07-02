import matplotlib.pyplot as plt
from botorch.acquisition.multi_objective.utils import is_non_dominated

def plot_pareto(train_y, trial, step, save_path=None, show_line=True):
    pareto_mask = is_non_dominated(train_y)
    pareto_front = train_y[pareto_mask]

    # 如果目标是最大化，BoTorch 的 hypervolume 默认视为最大化，所以不翻转目标
    # 若是最小化，请做：pareto_front = -pareto_front

    # 按第一个目标排序（使连线有序）
    pareto_front_sorted = pareto_front[pareto_front[:, 0].argsort()]

    plt.figure()
    plt.scatter(train_y[:, 0], train_y[:, 1], alpha=0.3, label='All Points')
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='Pareto Front')

    if show_line:
        plt.plot(pareto_front_sorted[:, 0], pareto_front_sorted[:, 1], color='red', linestyle='--', linewidth=1)

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title(f'Pareto Front - Trial {trial} Step {step}')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()