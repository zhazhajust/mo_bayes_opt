import torch
import pandas as pd
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

class BOLogger:
    def __init__(self, ref_point, save_models=False):
        self.ref_point = ref_point
        self.save_models = save_models
        self._history = []  # 每个 trial 的 list of step dicts
        self._current_trial = []

    def start_new_trial(self):
        if self._current_trial:
            self._history.append(self._current_trial)
        self._current_trial = []

    def log(self, train_x, train_y, next_x, next_y, models=None):
        # 超体积计算
        pareto_mask = is_non_dominated(train_y)
        pareto_y = train_y[pareto_mask]
        hv = Hypervolume(ref_point=self.ref_point)
        hypervolume = float(hv.compute(pareto_y))

        step_data = {
            'train_x': train_x.detach().clone(),
            'train_y': train_y.detach().clone(),
            'next_x': next_x.detach().clone(),
            'next_y': next_y.detach().clone(),
            'hypervolume': hypervolume,
        }

        if self.save_models and models is not None:
            step_data['models'] = models

        self._current_trial.append(step_data)
        return hypervolume

    def finalize(self):
        if self._current_trial:
            self._history.append(self._current_trial)
            self._current_trial = []

    def get_history(self):
        return self._history

    def get_hypervolumes(self):
        return torch.tensor([
            [step['hypervolume'] for step in trial]
            for trial in self._history
        ])

    def save_csv(self, filepath):
        records = []
        for t_idx, trial in enumerate(self._history):
            for i, step in enumerate(trial):
                records.append({
                    'trial': t_idx,
                    'step': i,
                    'hypervolume': step['hypervolume']
                })
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
