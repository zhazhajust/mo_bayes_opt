import pandas as pd
from .trial_logger import BOTrialLogger
from .hypervolume_tracker import HypervolumeTracker

class BOLogger:
    def __init__(self, ref_point, save_models=False, save_path=None):
        self.save_models = save_models
        self.save_path = save_path
        self.hv_tracker = HypervolumeTracker(ref_point)

        self.trials = []
        self.current_trial = None

        if self.save_path:
            self._initialize_csv()

    def _initialize_csv(self):
        df = pd.DataFrame(columns=["trial", "step", "hypervolume", "x", "y"])
        df.to_csv(self.save_path, index=False)

    def start_new_trial(self):
        if self.current_trial:
            self.trials.append(self.current_trial)
        self.current_trial = BOTrialLogger()

    def log(self, x, y, models=None):
        hv = self.hv_tracker.compute(y)
        self.current_trial.log_step(x, y, hv, models)

        if self.save_path:
            self._save_step_to_csv(x, y, hv)

        return hv

    def _save_step_to_csv(self, x, y, hypervolume):
        record = {
            "trial": len(self.trials),
            "step": len(self.current_trial.steps) - 1,
            "hypervolume": hypervolume,
            "x": x.cpu().numpy().flatten(),
            "y": y.cpu().numpy().flatten(),
        }
        df = pd.DataFrame([record])
        df.to_csv(self.save_path, mode="a", header=False, index=False)

    def finalize(self):
        if self.current_trial:
            self.trials.append(self.current_trial)
            self.current_trial = None

    def get_hypervolumes(self):
        return [
            [step["hypervolume"] for step in trial.get_steps()]
            for trial in self.trials
        ]

    def get_all_data(self):
        return [trial.get_steps() for trial in self.trials]
