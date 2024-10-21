from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from basic.enums import OptimizationType


class UnaryFunction(ABC):
    """
    Represents a unary function.
    The function should be continuous, defined everywhere and has local optimum analyzability.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def optimum_type(self) -> OptimizationType:
        """
        Return the type of the local optimum in analysis of the function.
        """
        pass

    @abstractmethod
    def evaluate(self, x: float) -> float:
        """
        Evaluate the function at a given point.
        """
        pass

    @abstractmethod
    def local_optimum(self, x: float) -> (float, float):
        """
        Return tuple of (local optimum position to point x on X-axis, and its fitness) .
        """
        pass

    def __call__(self, x: float) -> float:
        return self.evaluate(x)

    def plot(self, x_min: float, x_max: float):
        """
        Plot the function in the given range.
        Useful for checking the correctness of implementation of local_optimum().
        Example:
            r = RastriginFunc()
            r.plot(-5.12, 5.12)
        """
        x = np.linspace(x_min, x_max, 500)
        y = [self.evaluate(xi) for xi in x]
        opt_y = [self.local_optimum(xi)[1] for xi in x]
        plt.plot(x, y, label='Fitness')
        plt.plot(x, opt_y, label='Optimum', linestyle='--')
        plt.title('Unary Function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()