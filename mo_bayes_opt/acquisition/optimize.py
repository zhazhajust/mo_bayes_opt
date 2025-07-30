import torch
import warnings
from botorch.optim import optimize_acqf

# ========== 各策略定义 ==========

def get_acq_func_ALT_EI(train_x, train_y, models, ref_point, bounds, q):
    from botorch.acquisition import ExpectedImprovement
    idx = train_y.shape[0] % train_y.shape[1]
    best_f = train_y[:, idx].max()
    return ExpectedImprovement(model=models[idx], best_f=best_f, maximize=True)

def get_acq_func_EHVI(train_x, train_y, models, ref_point, bounds, q):
    from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning

    model_list = ModelListGP(*models)
    model_list.eval()
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)

    return ExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=ref_point.tolist(),
        partitioning=partitioning
    )

def get_acq_func_qEHVI(train_x, train_y, models, ref_point, bounds, q):
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
    from botorch.sampling import SobolQMCNormalSampler
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning

    model_list = ModelListGP(*models)
    model_list.eval()
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)

    return qLogExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    )

def get_acq_func_PAREGO(train_x, train_y, models, ref_point, bounds, q):
    from botorch.acquisition import ExpectedImprovement
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

    weights = torch.rand(train_y.shape[1], device=train_x.device)
    weights /= weights.sum()
    scalarization = get_chebyshev_scalarization(weights=weights, Y=train_y)
    scalarized_y = scalarization(train_y)

    model = models[0]
    best_f = scalarized_y.max()
    return ExpectedImprovement(model=model, best_f=best_f, maximize=True)

def get_acq_func_UCB(train_x, train_y, models, ref_point, bounds, q):
    from botorch.acquisition.analytic import UpperConfidenceBound
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

    beta = 0.1
    ucb_vals = []
    for model in models:
        ucb = UpperConfidenceBound(model=model, beta=beta)
        ucb_vals.append(ucb)

    # 使用 scalarization 合并多目标
    weights = torch.rand(len(models), device=train_x.device)
    weights /= weights.sum()

    def scalarized_ucb(X):
        vals = torch.stack([a(X) for a in ucb_vals], dim=-1)
        return (vals * weights).sum(dim=-1, keepdim=True)

    class ScalarizedUCB:
        def __call__(self, X):
            return scalarized_ucb(X)

    return ScalarizedUCB()

def get_acq_func_NPAREGO(train_x, train_y, models, ref_point, bounds, q):
    from botorch.acquisition import qExpectedImprovement
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

    weights = torch.rand(train_y.shape[1], device=train_x.device)
    weights /= weights.sum()
    scalarization = get_chebyshev_scalarization(weights=weights, Y=train_y)
    scalarized_y = scalarization(train_y)

    model = models[0]
    best_f = scalarized_y.max()
    return qExpectedImprovement(model=model, best_f=best_f, maximize=True)

# ========== 策略注册字典 ==========
STRATEGY_MAP = {
    "ALT_EI": get_acq_func_ALT_EI,
    "EHVI": get_acq_func_EHVI,
    "QEHVI": get_acq_func_qEHVI,
    "PAREGO": get_acq_func_PAREGO,
    "NPAREGO": get_acq_func_NPAREGO,
    "UCB": get_acq_func_UCB,
    "RANDOM": None,
}

# ========== 主函数 ==========
def optimize_step(train_x, train_y, models, ref_point, bounds, strategy="EHVI", **kwargs):
    strategy = strategy.upper()
    bounds = bounds.to(train_x.device)
    q = kwargs.get("q", 1)
    if strategy not in STRATEGY_MAP:
        raise NotImplementedError(f"Strategy '{strategy}' is not supported.")

    if strategy == "RANDOM":
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(q, train_x.shape[1], device=train_x.device)

    # 调用对应策略函数获取 acquisition function
    acq_func = STRATEGY_MAP[strategy](train_x, train_y, models, ref_point, bounds, q)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        next_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=q,
            num_restarts=10,
            raw_samples=64,
        )

    return next_x
