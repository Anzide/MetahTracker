import platform

import matplotlib as mpl

from algorithms.pso import PsoAlgorithm
from basic.enums import OptimizationType
from basic.landscape import DimSeparatedLandscape
from functions.cos_decay_func import CosineDecayFunc
from functions.parabola_func import RandomParabolaFunc
from functions.rastrigin_func import RastriginFunc


def auto_select_mpl_backend():
    """
    Select the appropriate backend for matplotlib according to the current OS.
    ONLY TESTED ON WINDOWS.
    """
    current_os = platform.system()
    if current_os == 'Windows':
        mpl.use('Qt5Agg')
    elif current_os == 'Linux':
        mpl.use('TkAgg')
    elif current_os == 'Darwin':
        mpl.use('MacOSX')
    else:
        raise RuntimeError(f"Unknown OS: {current_os}")


def test_1d():
    ls = DimSeparatedLandscape(1,
                               OptimizationType.MAXIMISATION,
                               [CosineDecayFunc()],
                               [(-20, 20)])
    # print(ls.compare_solution([0.2], [5]))
    # print(ls.compare_solution([5.7], [0.1]))
    ls.plot_1d_func()


def test_2d():
    ls = DimSeparatedLandscape(2,
                               OptimizationType.MAXIMISATION,
                               [CosineDecayFunc(1, 1), CosineDecayFunc(1, 1)],
                               [(-20, 20), (-20, 20)])
    # print(l2d.compare_solution([0.2, 0.2], [5, 5]))
    # print(ls.evaluate([2.81, -1.16]))
    ls.plot_2d_func_flat()


def test_pso_cos_2d():
    ls = DimSeparatedLandscape(2,
                               OptimizationType.MAXIMISATION,
                               [CosineDecayFunc(1, 1), CosineDecayFunc(1, 1)],
                               [(-20, 20), (-20, 20)])
    pso = PsoAlgorithm(ls, 2, 20, max_iter=10, seed=123)
    pso.run()
    # print(pso.results)
    # pso.print_results()
    ls.plot_2d_func_flat(frame_interval=1000, results=pso.results, show_dots=False)


def test_pso_rastrigin_2d():
    ls = DimSeparatedLandscape(2,
                               OptimizationType.MINIMISATION,
                               [RastriginFunc(), RastriginFunc()],
                               [(-5.12, 5.12), (-5.12, 5.12)])
    pso = PsoAlgorithm(ls, 2, 40, max_iter=15, seed=123)
    pso.run()
    ls.plot_2d_func_flat(frame_interval=1000, results=pso.results, show_dots=False)


def test_parabola_landscape():
    rpf1 = RandomParabolaFunc(max_a=-5, a_randomness=2, min_fitness=-10, max_fitness=10, left_boundary=-20,
                              right_boundary=20, spacing_coefficient=0.5)
    rpf2 = RandomParabolaFunc(max_a=-5, a_randomness=2, min_fitness=-10, max_fitness=10, left_boundary=-20,
                              right_boundary=20, spacing_coefficient=0.5)
    ls = DimSeparatedLandscape(2,
                               OptimizationType.MAXIMISATION,
                               [rpf1,rpf2],
                               [(-20, 20),(-20, 20)])
    # ls.plot_2d_func_surface()
    pso = PsoAlgorithm(ls, 2, 40, max_iter=20, seed=123)
    pso.run()
    ls.plot_2d_func_flat(frame_interval=1000, results=pso.results, show_dots=False)

def main():
    auto_select_mpl_backend()

    # show_custom_func()
    # test_1d()
    # test_2d()
    # test_pso_cos_2d()
    # test_pso_rastrigin_2d()
    test_parabola_landscape()


if __name__ == '__main__':
    main()
