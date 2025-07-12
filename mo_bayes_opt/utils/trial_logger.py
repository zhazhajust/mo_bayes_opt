class BOTrialLogger:
    def __init__(self):
        self.steps = []

    def log_step(self, x, y, hypervolume, models=None):
        self.steps.append({
            "x": x.detach().clone(),
            "y": y.detach().clone(),
            "hypervolume": hypervolume,
            "models": models if models is not None else None
        })

    def get_steps(self):
        return self.steps
