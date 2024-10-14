import math

from basic.enums import OptimizationType
from basic.function import UnaryFunction


class CosineFunc(UnaryFunction):

    def __init__(self, a: float, w: float):
        super().__init__()
        self.a = a
        self.w = w
        self.t = 2 * math.pi / w

    def optimum_type(self) -> OptimizationType:
        return OptimizationType.MAXIMISATION

    def evaluate(self, x: float) -> float:
        return self.a * math.cos(self.w * x)

    def local_optimum(self, x: float) -> (float, float):
        return 2 * math.pi / self.w, self.a
