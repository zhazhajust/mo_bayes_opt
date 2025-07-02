# 🎯 Multi-Objective Bayesian Optimization (MOBO)

A modular Python library for **multi-objective Bayesian optimization** using **Gaussian Processes (GP)** and **BoTorch**, with optional support for **adaptive noise modeling**.

> Built with [PyTorch](https://pytorch.org/), [GPyTorch](https://gpytorch.ai/), and [BoTorch](https://botorch.org/)

---

## 📦 Features

- ✔️ Multi-objective optimization with EHVI / ALT-EI
- ✔️ Independent GP models per objective
- ✔️ Adaptive noise learning using a neural `NoiseNet`
- ✔️ Pareto front extraction and visualization
- ✔️ Hypervolume logging per step
- ✔️ Easy-to-extend architecture for experiments

---

## 📁 Project Structure

mo_bayes_opt/
├── models/ # GPModel, NoiseNet, GPTrainer
├── acquisition/ # Acquisition function optimization
├── core/ # BO loop & objective functions
├── utils/ # Logging, visualization
├── experiments/ # Example experiments
├── data/ # Hypervolume log storage

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Example Optimization

```python

def joint_objective(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    interaction = torch.sin(5 * torch.pi * x1 * x2)
    f1 = ((x[:, :3] * torch.sin(3 * torch.pi * x[:, :3])).sum(dim=-1) + 0.5 * interaction)
    f2 = (((1 - x[:, 3:]) * torch.cos(3 * torch.pi * x[:, 3:])).sum(dim=-1) - 0.3 * interaction)
    return torch.stack([f1, f2], dim=-1)

bo = MultiObjectiveBO(
    objective_fn=joint_objective,
    input_dim=3,
    bounds=torch.tensor([[0.0] * 3, [1.0] * 3]),
    ref_point=torch.tensor([0.0, 0.0]),
    strategy="EHVI",
    use_adaptive_noise=False
)
hypervolumes = bo.run(num_repeats=1, num_queries=100)
```

### 3. Output
Console: Logs trial progress and hypervolume

File: data/hypervolume_log.csv containing hypervolume progression

### 📈 Visualization
Use plot_pareto(train_y, trial, step) to visualize the Pareto front at a given step.
