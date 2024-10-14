from typing import Callable, Tuple

import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import ndarray

from basic.enums import exploration_color_map, OptimizationType, OptimumComparison
from basic.function import UnaryFunction
from basic.enums import MoveType


class Landscape:
    """
    A landscape is a function that maps a point in the search space to a fitness value.
    The Landscape class takes any kind of function, and is NOT local optimum analysable.
    """

    def __init__(self,
                 dim: int,
                 optimization_type: OptimizationType,
                 evaluator: Callable[[ndarray], float],
                 bounds: list[Tuple[float, float]]):
        """
        :param dim: The number of dimensions.
        :param optimization_type: The type of optimization.
        :param evaluator: The fitness function that takes a ndarray of shape (dim) and returns fitness in float.
        :param bounds: An array of tuples, each tuple represents the lower and upper bound of the corresponding dimension.
        """
        self.dim = dim
        self.optimization_type = optimization_type
        self.evaluator = evaluator
        self.bounds = bounds

    def is_in_bounds(self, x: ndarray) -> bool:
        for i, bound in enumerate(self.bounds):
            if not bound[0] <= x[i] <= bound[1]:
                return False
        return True

    def evaluate(self, x: ndarray) -> float:
        return self.evaluator(x)

    def plot_1d_func(self, point_num: int = 500):
        if self.dim != 1:
            raise ValueError("This function should only be used for drawing 1D functions.")

        x = np.linspace(self.bounds[0][0], self.bounds[0][1], point_num)
        y = [self.evaluate(np.array([xi])) for xi in x]
        plt.plot(x, y, label='Fitness')
        plt.title('Function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def plot_2d_func_flat(self, show_dots: bool = False, show_figures: bool = True, frame_interval: int = 1000,
                          results=None):
        """
        Plot the 2D function in a flat style.
        This function has the ability to ANIMATE the exploration process, and therefore is the most important exhibit function.
        :param show_dots: Use dots to display an individual.
        :param show_figures: Use figures to display an individual.
        :param frame_interval: The interval between animation frames in milliseconds.
        :param results: Used for animation purpose only.
        A ndarray of shape (frame_num, point_num) where each element (which represents a single frame) is a tuple of (point: ndarray, move_type: MoveType).
        :return:
        """
        if self.dim != 2:
            raise ValueError("This function should only be used for drawing 2-dim functions.")

        x = np.linspace(self.bounds[0][0], self.bounds[0][1], 500)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], 500)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                coordinate = np.array([X[i, j], Y[i, j]])
                Z[i, j] = self.evaluate(coordinate)

        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.75)
        ax.legend("test legend")
        plt.colorbar(contour, label='Value')
        plt.title('2D Func')
        plt.xlabel('x')
        plt.ylabel('y')

        # TODO: Some points might go out of bounds
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])

        # for name,_ in OptimizationType.__members__.items():

        # red_patch = mpatches.Patch(color='red', label='Important Area')

        patches = [mpatches.Patch(color=color, label=name.value) for name, color in exploration_color_map.items()]
        ax.legend(handles=patches, loc='upper right')

        fig.tight_layout()

        if results is not None:

            # Save the scatter object to remove it later
            scatter = None
            texts = []

            def update(frame):
                nonlocal scatter, texts

                # Remove objects of last frame
                if scatter:
                    scatter.remove()
                for text in texts:
                    text.remove()
                texts = []

                # Shape: (num, 2)
                particle_tuples = results[frame]
                particle_pos = np.array([p[0] for p in particle_tuples])
                colors = [exploration_color_map[p[1]] for p in particle_tuples]

                if show_dots:
                    scatter = ax.scatter(particle_pos[:, 0], particle_pos[:, 1], color=colors, s=5, zorder=10)

                if show_figures:
                    for i, (x, y) in enumerate(particle_pos):
                        text = ax.text(x, y, str(i), color=colors[i], fontsize=8, ha='center', va='center',
                                       zorder=11)
                        texts.append(text)

            ani = FuncAnimation(fig, update, frames=len(results), interval=frame_interval, repeat=False)

        plt.show()

    def plot_2d_func_surface(self):
        """
        Plot the landscape in a 3D surface style.
        """

        if self.dim != 2:
            raise ValueError("This function should only be used for drawing 2D functions.")

        x = np.linspace(self.bounds[0][0], self.bounds[0][1], 500)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], 500)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                coordinate = np.array([X[i, j], Y[i, j]])
                Z[i, j] = self.evaluate(coordinate)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


class DimSeparatedLandscape(Landscape):

    def __init__(self, dim: int,
                 global_target_type: OptimizationType,
                 funcs: list[UnaryFunction],
                 bounds: list[Tuple[float, float]]):

        self.funcs = funcs
        if len(funcs) != dim:
            raise ValueError("The number of functions should be equal to the dimension.")

        for f in funcs:
            if f.optimum_type() != global_target_type:
                raise ValueError("The optimum type of all functions should be the same as the global target type.")

        def evaluate_fitness(x: ndarray) -> float:
            sum_value: float = 0
            for i, v in enumerate(x):
                sum_value += funcs[i].evaluate(v)
            return sum_value

        super().__init__(dim, global_target_type, evaluate_fitness, bounds)

    def compare_local_optimum(self, base: ndarray, target: ndarray) -> OptimumComparison:
        """
        Return how the target local optimum compares to the base local optimum. BETTER if target is better than base.
        """
        base_optimum_coord = []
        target_optimum_coord = []
        base_optimum_fitness = 0
        target_optimum_fitness = 0

        # Sum the fitness of the functions on each dim
        for i, f in enumerate(self.funcs):
            b_coord, b_fit = f.local_optimum(base[i])
            t_coord, t_fit = f.local_optimum(target[i])

            base_optimum_coord.append(b_coord)
            target_optimum_coord.append(t_coord)
            base_optimum_fitness += b_fit
            target_optimum_fitness += t_fit

        if np.allclose(base_optimum_coord, target_optimum_coord, rtol=1e-9, atol=0.0):
            return OptimumComparison.SAME

        # base_optimum_coord = np.array(base_optimum_coord)
        # target_optimum_coord = np.array(target_optimum_coord)

        # print("--- compare_local_optimum START ---")
        # print(f"base coord: {base_optimum_coord}, base fitness: {base_optimum_fitness}")
        # print(f"target coord: {target_optimum_coord}, target fitness: {target_optimum_fitness}")
        # print("--- compare_local_optimum END   ---")
        if self.optimization_type == OptimizationType.MAXIMISATION:
            return OptimumComparison.BETTER if target_optimum_fitness > base_optimum_fitness else OptimumComparison.WORSE
        else:
            return OptimumComparison.BETTER if target_optimum_fitness < base_optimum_fitness else OptimumComparison.WORSE

    # TODO: Not yet consider the situation where fitness is equal.
    def compare_solution(self, base: ndarray, target: ndarray) -> MoveType:
        """
        Compare two solutions on the landscape, and return how the target solution compares to the base solution.
        """
        optimum_comparison = self.compare_local_optimum(base, target)
        if optimum_comparison == OptimumComparison.SAME:
            return MoveType.EXPLOITATION

        base_fitness = self.evaluate(base)
        target_fitness = self.evaluate(target)
        if self.optimization_type == OptimizationType.MAXIMISATION:
            fitness_comparison = target_fitness > base_fitness
        else:
            fitness_comparison = target_fitness < base_fitness

        # print("--- compare_solution START ---")
        # print(f"base: {base}, target: {target}, base_fitness: {base_fitness}, target_fitness: {target_fitness}")
        # print(f"base_fitness: {base_fitness}, target_fitness: {target_fitness}")
        # print("--- compare_solution END   ---")

        if optimum_comparison == OptimumComparison.BETTER and fitness_comparison:
            return MoveType.TRUE_EXPL
        elif optimum_comparison == OptimumComparison.WORSE and not fitness_comparison:
            return MoveType.TRUE_REJ
        elif optimum_comparison == OptimumComparison.WORSE and fitness_comparison:
            return MoveType.FALSE_EXPL
        elif optimum_comparison == OptimumComparison.BETTER and not fitness_comparison:
            return MoveType.FALSE_REJ
        else:
            raise ValueError("Unknown comparison between base and target.")
