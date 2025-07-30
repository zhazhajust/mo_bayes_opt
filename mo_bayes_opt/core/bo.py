import torch
from tqdm import tqdm
from mo_bayes_opt.models.gp import train_gp_models
from mo_bayes_opt.acquisition.optimize import optimize_step
from mo_bayes_opt.utils.bo_logger import BOLogger  # 使用新 logger

class MultiObjectiveBO:
    def __init__(
        self,
        objective_fn,
        input_dim=6,
        bounds=None,
        ref_point=None,
        strategy="EHVI",
        use_adaptive_noise=True,
        num_iters=500,
        save_models=False,
        initial_data=None,  # Tuple (x, y)
    ):
        self.objective_fn = objective_fn
        self.input_dim = input_dim
        self.bounds = bounds if bounds is not None else torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        self.ref_point = ref_point if ref_point is not None else torch.tensor([-0.5 * input_dim] * 2)
        self.strategy = strategy
        self.use_adaptive_noise = use_adaptive_noise
        self.num_iters = num_iters
        self.save_models = save_models
        self.initial_data = initial_data

    def run(self, num_repeats=3, num_queries=20, log_path=None, **kwargs):
        self.logger = BOLogger(ref_point=self.ref_point, save_models=self.save_models, save_path=log_path)

        for trial in range(num_repeats):
            print(f"\nTrial {trial}")
            torch.manual_seed(trial)

            if self.initial_data is not None:
                train_x = self.initial_data[0].clone()
                train_y = self.initial_data[1].clone()
            else:
                train_x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(1, self.input_dim)
                train_y = self.objective_fn(train_x)

            self.logger.start_new_trial()
            self.logger.log(train_x, train_y)

            for _ in tqdm(range(num_queries)):
                models, _ = train_gp_models(
                    train_x, train_y, use_adaptive_noise=self.use_adaptive_noise, num_iters=self.num_iters
                )
                next_x = optimize_step(train_x, train_y, models, self.ref_point, self.bounds, self.strategy, **kwargs)
                next_y = self.objective_fn(next_x)

                train_x = torch.cat([train_x, next_x])
                train_y = torch.cat([train_y, next_y])

                self.logger.log(next_x, next_y, models)

        self.logger.finalize()
        return self.logger.get_hypervolumes()
