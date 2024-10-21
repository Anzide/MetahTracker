from algorithms.pso import PsoAlgorithm
from basic.enums import OptimizationType
from basic.landscape import DimSeparatedLandscape
from basic.utils import auto_select_mpl_backend
from functions.cos_decay_func import CosineDecayFunc
from functions.parabola_func import RandomParabolaFunc
from functions.rastrigin_func import RastriginFunc


def choose_rastrigin_landscape():
    return DimSeparatedLandscape(2,
                                 OptimizationType.MINIMISATION,
                                 [RastriginFunc(), RastriginFunc()],
                                 [(-5.12, 5.12), (-5.12, 5.12)])


def choose_decaying_cosine_landscape():
    return DimSeparatedLandscape(2,
                                 OptimizationType.MAXIMISATION,
                                 [CosineDecayFunc(1, 1), CosineDecayFunc(1, 1)],
                                 [(-20, 20), (-20, 20)])


def choose_random_parabola_landscape():
    rpf1 = RandomParabolaFunc(max_a=-5, a_randomness=2, min_fitness=-10, max_fitness=10, left_boundary=-20,
                              right_boundary=20, spacing_coefficient=0.5)
    rpf2 = RandomParabolaFunc(max_a=-5, a_randomness=2, min_fitness=-10, max_fitness=10, left_boundary=-20,
                              right_boundary=20, spacing_coefficient=0.5)
    return DimSeparatedLandscape(2,
                                 OptimizationType.MAXIMISATION,
                                 [rpf1, rpf2],
                                 [(-20, 20), (-20, 20)])


def main():
    auto_select_mpl_backend()

    # Choose the landscape

    # ls = choose_decaying_cosine_landscape()
    # ls = choose_random_parabola_landscape()
    ls = choose_rastrigin_landscape()

    # Show the landscape
    ls.plot_2d_func_surface()

    # After closing the plot, create and run PSO algorithm
    pso = PsoAlgorithm(ls, dim=2, particle_num=40, max_iter=20, seed=123)
    pso.run()

    # Show the exploration progress
    ls.plot_2d_exploration(frame_interval=1000, results=pso.results)


if __name__ == '__main__':
    main()
