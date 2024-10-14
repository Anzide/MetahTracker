import numpy as np
from numpy import ndarray

from basic.enums import OptimizationType, MoveType
from basic.algorithm import Algorithm
from basic.landscape import DimSeparatedLandscape


class PsoAlgorithm(Algorithm):

    # TODO: This PSO implementation does not consider the boundary of the landscape.

    def __init__(self, landscape: DimSeparatedLandscape, dim: int, particle_num: int = 30, max_iter: int = 100,
                 w: float = 0.5, c1: float = 1, c2: float = 2,
                 seed: int = None):
        super().__init__(landscape)
        self.dim = dim
        self.particle_num = particle_num
        self.iter_num = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        # Results Shape: (max_iter+1, particle_num)
        self.results: ndarray = None
        self.update_counter = 0
        self.seed = seed

    def run(self, debug=False):

        np.random.seed(self.seed)
        num = self.particle_num
        dim = self.dim
        # Particles Shape: (num, dim)
        particles = np.array([np.random.uniform(low, high, size=num) for (low, high) in self.landscape.bounds]).T
        velocities = np.zeros((num, dim))
        best_positions = np.copy(particles)
        current_fitness = np.array([self.landscape.evaluate(p) for p in particles])

        if self.landscape.optimization_type == OptimizationType.MAXIMISATION:
            swarm_best_position = particles[np.argmax(current_fitness)]
            swarm_best_fitness = np.max(current_fitness)
        else:
            swarm_best_position = particles[np.argmin(current_fitness)]
            swarm_best_fitness = np.min(current_fitness)

        self.results = np.empty((self.iter_num + 1, num), dtype=object)
        for i in range(num):
            self.results[0, i] = (particles[i], MoveType.EXPLOITATION)

        for curr_iter in range(self.iter_num):
            # Update velocities
            r1 = np.random.uniform(0, 1, (num, dim))
            r2 = np.random.uniform(0, 1, (num, dim))
            velocities = self.w * velocities + self.c1 * r1 * (best_positions - particles) + self.c2 * r2 * (
                    swarm_best_position - particles)

            # Update positions
            new_particles = particles + velocities
            for i in range(num):
                self.results[curr_iter + 1, i] = (
                    new_particles[i], self.landscape.compare_solution(particles[i], new_particles[i]))
            particles = new_particles
            new_fitness = np.array([self.landscape.evaluate(p) for p in particles])

            # Update best positions and best fitness
            if self.landscape.optimization_type == OptimizationType.MAXIMISATION:
                improved_indices = np.where(new_fitness > current_fitness)
            else:
                improved_indices = np.where(new_fitness < current_fitness)
            best_positions[improved_indices] = particles[improved_indices]
            current_fitness[improved_indices] = new_fitness[improved_indices]

            if (self.landscape.optimization_type == OptimizationType.MAXIMISATION and
                    np.max(new_fitness) > swarm_best_fitness):
                self.update_counter = self.update_counter + 1
                swarm_best_position = particles[np.argmax(new_fitness)]
                swarm_best_fitness = np.max(new_fitness)
            elif (self.landscape.optimization_type == OptimizationType.MINIMISATION and
                  np.min(new_fitness) < swarm_best_fitness):
                self.update_counter = self.update_counter + 1
                swarm_best_position = particles[np.argmin(new_fitness)]
                swarm_best_fitness = np.min(new_fitness)

        print('PSO Best position: ', swarm_best_position)
        print('PSO Best fitness: ', swarm_best_fitness)

    def print_results(self):
        """
        Used for debugging.
        """
        for frame in range(self.results.shape[0]):
            print(f"--- Frame {frame} ---")
            for i in range(self.results.shape[1]):
                print(
                    f"P {i} - Pos: {self.results[frame, i][0]} , Fit: {self.landscape.evaluate(self.results[frame, i][0])} Move: {self.results[frame, i][1]}")
            print("\n")
