好的！下面是你现有 `README.md` 加上对新策略支持的完整修改版本，**已合并原有内容并补充了新特性说明和使用方法**。

---

```markdown
# 🎯 Multi-Objective Bayesian Optimization (MOBO)

A modular Python library for **multi-objective Bayesian optimization** using **Gaussian Processes (GP)** and **BoTorch**, with optional support for **adaptive noise modeling**.

> Built with [PyTorch](https://pytorch.org/), [GPyTorch](https://gpytorch.ai/), and [BoTorch](https://botorch.org/)

---

## 📦 Features

- ✔️ Multi-objective optimization with EHVI / ALT-EI / QEHVI / PAREGO / NPAREGO / UCB
- ✔️ Strategy selection via `strategy=` argument
- ✔️ Independent GP models per objective
- ✔️ Adaptive noise learning using a neural `NoiseNet`
- ✔️ Pareto front extraction and visualization
- ✔️ Hypervolume logging per step
- ✔️ Easy-to-extend architecture for experiments

---

## 🆕 What's New

- ✅ **Added support for more acquisition strategies**:
  - `EHVI` – Expected Hypervolume Improvement
  - `QEHVI` – Batch Expected Hypervolume Improvement
  - `ALT_EI` – Alternating Expected Improvement
  - `PAREGO` – Scalarization-based using Chebyshev method
  - `NPAREGO` – q-Expected Improvement with scalarization
  - `UCB` – Upper Confidence Bound with scalarization
  - `RANDOM` – Random uniform sampling baseline
- ✅ Strategy controlled via `strategy=` in `MultiObjectiveBO`
- ✅ Modular acquisition function registration using `STRATEGY_MAP`

---

## 📁 Project Structure

```

mo\_bayes\_opt/
├── models/         # GPModel, NoiseNet, GPTrainer
├── acquisition/    # Acquisition function optimization (strategies registered here)
├── core/           # BO loop & objective functions
├── utils/          # Logging, visualization
├── experiments/    # Example experiments
├── data/           # Hypervolume log storage

````

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

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
    strategy="PAREGO",  # 👈 Choose any supported strategy here
    use_adaptive_noise=False
)
hypervolumes = bo.run(num_repeats=1, num_queries=100)
```

### 3. Output

* 📊 Console: Logs optimization progress and hypervolume
* 📁 File: `data/hypervolume_log.csv` containing hypervolume progression

---

## ⚙️ Supported Strategies

| Strategy  | Description                                |
| --------- | ------------------------------------------ |
| `EHVI`    | Expected Hypervolume Improvement           |
| `QEHVI`   | Batch EHVI (q > 1)                         |
| `ALT_EI`  | Alternating EI over each objective         |
| `PAREGO`  | Scalarized EI with Chebyshev scalarization |
| `NPAREGO` | q-EI with scalarization                    |
| `UCB`     | Scalarized Upper Confidence Bound          |
| `RANDOM`  | Random uniform sampling                    |

---

## 🧩 Adding Your Own Strategy

To add a custom acquisition strategy:

1. Define your function in `acquisition/`:

```python
def get_acq_func_MY_STRATEGY(...):
    # return your acquisition function
```

2. Register it in `STRATEGY_MAP`:

```python
STRATEGY_MAP = {
    ...
    "MY_STRATEGY": get_acq_func_MY_STRATEGY,
}
```

3. Use it like:

```python
bo = MultiObjectiveBO(
    ...,
    strategy="MY_STRATEGY",
)
```

---

## 📈 Visualization

To visualize the Pareto front at a given step:

```python
plot_pareto(train_y, trial, step)
```

---

## 🧪 Experiments

Example scripts are in `experiments/`. You can run batch or scalarized optimization, compare strategies, and log results.

---

## 📬 Contact

If you encounter bugs or have suggestions, feel free to open an issue or contribute!

---

````
