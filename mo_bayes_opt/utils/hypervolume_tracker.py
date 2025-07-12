from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

class HypervolumeTracker:
    def __init__(self, ref_point):
        self.ref_point = ref_point
        self.hv = Hypervolume(ref_point)

    def compute(self, Y):
        pareto_mask = is_non_dominated(Y)
        pareto_front = Y[pareto_mask]
        return float(self.hv.compute(pareto_front))
