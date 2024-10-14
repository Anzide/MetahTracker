import math

from basic.enums import OptimizationType
from basic.function import UnaryFunction


class RastriginFunc(UnaryFunction):

    def __init__(self, A: float = 10):
        super().__init__()
        self.A = A

    def optimum_type(self) -> OptimizationType:
        return OptimizationType.MINIMISATION

    def evaluate(self, x: float) -> float:
        return self.A + (x ** 2 - self.A * math.cos(2 * math.pi * x))

    def local_optimum(self, x: float) -> (float, float):
        return round(x), self.evaluate(round(x))
