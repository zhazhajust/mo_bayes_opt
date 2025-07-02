import warnings
from botorch.models import ModelListGP
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.utils import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.optim import optimize_acqf

def optimize_step(train_x, train_y, models, ref_point, bounds, strategy="EHVI"):
    strategy = strategy.upper()
    
    if strategy == "ALT_EI":
        idx = train_y.shape[0] % 2
        best_f = train_y[:, idx].max()
        acq_func = ExpectedImprovement(model=models[idx], best_f=best_f, maximize=True)
    else:
        model_list = ModelListGP(*models)
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
        acq_func = ExpectedHypervolumeImprovement(
            model=model_list, ref_point=ref_point.tolist(), partitioning=partitioning
        )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        next_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=100
        )
    return next_x
