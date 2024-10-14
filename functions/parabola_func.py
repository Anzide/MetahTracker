import math
import random
from abc import ABC
from typing import Tuple, List

import numpy as np

from basic.enums import OptimizationType
from basic.function import UnaryFunction


class Parabola:
    def __init__(self, vertex: np.ndarray, a: float):
        self.vertex = vertex  # numpy array [x0, y0]
        self.a = a

    def __str__(self):
        return f"vertex: ({self.vertex[0]}, {self.vertex[1]}), a: {self.a}"

    def evaluate(self, x: float) -> float:
        x0, y0 = self.vertex
        return self.a * (x - x0) ** 2 + y0

    def solve_y(self, y: float) -> Tuple[float, float]:
        """
        Solves for x in the equation y = a*(x - x0)^2 + y0.
        Returns a tuple of two x-values where the parabola intersects the given y.
        If a = 0 or there are no real solutions, throw Error.
        Guarantees x1 < x2.
        """
        x0, y0 = self.vertex
        a = self.a

        if a == 0:
            raise ValueError("The coefficient 'a' must not be zero.")

        discriminant = (y - y0) / a
        if discriminant < 0:
            raise ValueError("No real solutions exist for the given y-value.")
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            x1 = x0 + sqrt_discriminant
            x2 = x0 - sqrt_discriminant
            if x1 > x2:
                return x2, x1
            else:
                return x1, x2


def solve_a_for_parabola(vertex: np.ndarray, point: np.ndarray) -> float:
    """
    Solves for the coefficient 'a' such that the parabola with the given vertex passes through the given point.
    Return value is the Parabola.
    """
    x0, y0 = vertex
    x, y = point

    if x == x0:
        raise ValueError("The x-value of the point must be different from the vertex x-value to solve for 'a'.")

    a = (y - y0) / ((x - x0) ** 2)
    return a


class RandomParabolaFunc(UnaryFunction, ABC):
    def __init__(
            self,
            max_a: float = -1.0,
            a_randomness: float = 2.0,
            min_fitness: float = 0,
            max_fitness: float = 1.0,
            left_boundary: float = 0,
            right_boundary: float = 10,
            spacing_coefficient: float = 0.0,
            seed: int = None
    ):
        """
        :param max_a: The maximum 'a' coefficient for the parabolas. Must be negative.
        An overly small value may result in overly crowded. An overly large value may result in overly sparse.
        :param a_randomness: Must be greater than 1. Greater values result in more randomness in the 'a' coefficient.
        Overly large values may result in overly crowded.
        :param min_fitness: The minimum fitness value for the function.
        :param max_fitness: The maximum fitness value for the function.
        :param spacing_coefficient: The coefficient to determine the spacing between the parabolas. Must be in the range (0, 1).
        Greater values result in more spacing. Recommended values greater than 0.1 to prevent overly crowded.
        """
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        if max_a >= 0:
            raise ValueError("The maximum 'a' coefficient must be negative.")
        self.max_a = max_a
        self.a_random_coefficient = a_randomness
        self.min_fitness = min_fitness
        self.max_fitness = max_fitness
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        self.parabolas: List[Parabola] = []
        first_p = Parabola(np.array([left_boundary, random.uniform(min_fitness, max_fitness)]),
                           random.uniform(max_a, 0))
        self.parabolas.append(first_p)

        last_p = first_p

        while True:
            # vertex_x = random.uniform(last_p.vertex[0], last_p.solve_y(min_height)[1])
            last_x = last_p.vertex[0]
            max_x = last_p.solve_y(min_fitness)[1]
            vertex_x = random.uniform(last_x + (max_x - last_x) * spacing_coefficient, max_x)
            vertex_y = random.uniform(last_p.evaluate(vertex_x), max_fitness)

            if vertex_y < last_p.vertex[1]:
                a = random.uniform(a_randomness * max_a, max_a)
            else:
                intersect_a = solve_a_for_parabola([vertex_x, vertex_y], last_p.vertex)
                # Should not happen
                if intersect_a > 0:
                    raise ValueError("The minimum 'a' coefficient must be negative.")
                temp_max_a = min(max_a, intersect_a)
                a = random.uniform(a_randomness * temp_max_a, temp_max_a)

            p = Parabola(np.array([vertex_x, vertex_y]), a)
            self.parabolas.append(p)
            last_p = p
            if p.evaluate(right_boundary) > min_fitness:
                break

        # Convert parabolas to NumPy arrays for efficient access
        self.vertices = np.array([p.vertex for p in self.parabolas])  # Shape: (n, 2)
        self.a_coeffs = np.array([p.a for p in self.parabolas])  # Shape: (n,)

        self.check()

    def optimum_type(self) -> OptimizationType:
        return OptimizationType.MAXIMISATION

    def evaluate(self, x: float) -> float:
        l, r = self.binary_find_index(x)
        if l == -1:
            return self.parabolas[r].evaluate(x)
        if r == -1:
            return self.parabolas[l].evaluate(x)
        return max(self.parabolas[l].evaluate(x), self.parabolas[r].evaluate(x))

    def local_optimum(self, x: float) -> Tuple[float, float]:
        l, r = self.binary_find_index(x)
        if l == -1:
            x = self.parabolas[r].vertex[0]
        elif r == -1:
            x = self.parabolas[l].vertex[0]
        else:
            if self.parabolas[l].evaluate(x) > self.parabolas[r].evaluate(x):
                x = self.parabolas[l].vertex[0]
            else:
                x = self.parabolas[r].vertex[0]
        return x, self.evaluate(x)

    def binary_find_index(self, x: float) -> Tuple[int, int]:
        # Binary search to find the index of the parabola whose vertex x-coordinate is just greater than 'x'
        arr = self.vertices[:, 0]
        left, right = 0, len(arr) - 1

        if x < arr[left]:
            return -1, left
        if x > arr[right]:
            return right, -1

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == x:
                return mid, mid
            if arr[mid] < x < arr[mid + 1]:
                return mid, mid + 1
            if arr[mid] < x:
                left = mid + 1
            else:
                right = mid - 1

        raise ValueError("Binary search failed to find the index.")

    def check(self) -> bool:
        """
        Check if no parabola overlaps the vertex of a nearby parabola. If it does, raise an error.
        :return:
        """
        for i in range(1, len(self.parabolas) - 1):
            if self.parabolas[i].evaluate(self.parabolas[i - 1].vertex[0]) > self.parabolas[i].vertex[1]:
                raise ValueError(f"Parabola {i} violates the condition.")
        return True
