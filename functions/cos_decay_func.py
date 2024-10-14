import math

from basic.enums import OptimizationType
from basic.function import UnaryFunction


def get_period_id(x: float, period: float, start_at: float = 0) -> int:
    """
    Return the id of the period that x is in. Could be Negative. Period 0 starts at start_at.
    """
    return math.floor((x - start_at) / period)


def get_nearest_period_start(x: float, period: float, start_at: float = 0):
    """
    Return the nearest period start position. NOT the start of the current period!
    If x is closer to the start of the next period, this will return the start of the next period.
    """
    return round((x - start_at) / period) * period + start_at


class CosineDecayFunc(UnaryFunction):
    """
    Like a rastrigin function but has local minimum instead of local maximum.
    """
    def __init__(self, amplitude: float = 1.0, angular_frequency: float = 1.0, decay_rate: float = 0.1):
        super().__init__()
        self.a = amplitude
        self.w = angular_frequency
        self.t = 2 * math.pi / angular_frequency
        self.d = decay_rate

    def optimum_type(self) -> OptimizationType:
        return OptimizationType.MAXIMISATION

    # fitness = a * cos(w * x) * e ^ (-d * |k|)
    def evaluate(self, x: float) -> float:
        return self.a * math.cos(self.w * x) * math.exp(
            -self.d * math.fabs(get_period_id(x, self.t / 2, -self.t / 4)))

    def local_optimum(self, x: float) -> (float, float):
        pos = get_nearest_period_start(x, self.t)
        return pos, self.evaluate(pos)