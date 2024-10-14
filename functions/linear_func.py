import math

from basic.enums import OptimizationType
from basic.function import UnaryFunction


class LinearFunc(UnaryFunction):
    # TODO: Not well tested yet
    def __init__(self, weight: float):
        super().__init__()
        if math.isclose(weight, 0):
            raise ValueError('Weight should not be zero')
        self.weight = weight

    def optimum_type(self) -> OptimizationType:
        return OptimizationType.MAXIMISATION

    def evaluate(self, x: float) -> float:
        return x * self.weight

    def local_optimum(self, x: float) -> (float, float):
        if self.weight > 0:
            return float('inf'), float('inf')
        else:
            return float('-inf'), float('inf')
