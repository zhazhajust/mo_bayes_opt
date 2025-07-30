import pandas as pd
import torch
import os

from .trial_logger import BOTrialLogger
from .hypervolume_tracker import HypervolumeTracker


class BOLogger:
    def __init__(self, ref_point, save_models=False, save_path=None):
        self.save_models = save_models
        self.save_path = save_path
        self.hv_tracker = HypervolumeTracker(ref_point)

        self.trials = []
        self.current_trial = None

        # Auto-detect format from file extension
        if self.save_path:
            if save_path.endswith('.h5') or save_path.endswith('.hdf5'):
                self.save_format = 'hdf5'
                import h5py
                self.h5file = h5py.File(self.save_path, 'w')
                self.h5_trials_group = self.h5file.create_group('trials')
            elif save_path.endswith('.csv'):
                self.save_format = 'csv'
                self._initialize_csv()
            else:
                raise ValueError("Unsupported file extension. Use '.csv' or '.h5/.hdf5'.")
        else:
            self.save_format = None

    def _initialize_csv(self):
        df = pd.DataFrame(columns=["trial", "step", "hypervolume", "x", "y"])
        df.to_csv(self.save_path, index=False)

    def start_new_trial(self):
        if self.current_trial:
            self.trials.append(self.current_trial)
        self.current_trial = BOTrialLogger()

    def log(self, x: torch.Tensor, y: torch.Tensor, models=None):
        hv = self.hv_tracker.compute(y)
        self.current_trial.log_step(x, y, hv, models)

        if self.save_path:
            if self.save_format == 'csv':
                self._save_step_to_csv(x, y, hv)
            elif self.save_format == 'hdf5':
                self._save_step_to_hdf5(x, y, hv)

        return hv

    def _save_step_to_csv(self, x, y, hypervolume):
        record = {
            "trial": len(self.trials),
            "step": len(self.current_trial.steps) - 1,
            "hypervolume": hypervolume,
            "x": x.cpu().numpy().tolist(),
            "y": y.cpu().numpy().tolist(),
        }
        df = pd.DataFrame([record])
        df.to_csv(self.save_path, mode="a", header=False, index=False)

    def _save_step_to_hdf5(self, x, y, hypervolume):
        import h5py

        trial_id = str(len(self.trials))
        step_id = str(len(self.current_trial.steps) - 1)

        trial_group = self.h5_trials_group.require_group(trial_id)
        step_group = trial_group.create_group(step_id)

        step_group.create_dataset("x", data=x.cpu().numpy())
        step_group.create_dataset("y", data=y.cpu().numpy())
        step_group.attrs["hypervolume"] = hypervolume

    def finalize(self):
        if self.current_trial:
            self.trials.append(self.current_trial)
            self.current_trial = None

        if self.save_format == 'hdf5' and hasattr(self, 'h5file'):
            self.h5file.close()

    def get_hypervolumes(self):
        return [
            [step["hypervolume"] for step in trial.get_steps()]
            for trial in self.trials
        ]

    def get_all_data(self):
        return [trial.get_steps() for trial in self.trials]
