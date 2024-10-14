import numpy as np
import matplotlib.pyplot as plt

# Define the 1D Rastrigin function
def rastrigin(x):
    return 10 + (x ** 2 - 10 * np.cos(2 * np.pi * x))

# Define the PSO algorithm for 1D with particle trajectories
def pso_1d_trajectories(cost_func, dim=1, num_particles=5, max_iter=10, w=0.5, c1=1, c2=2):
    # Initialize particles and velocities
    particles = np.random.uniform(-5.12, 5.12, num_particles)
    velocities = np.zeros(num_particles)

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # To store particle trajectories
    particle_trajectories = [np.copy(particles)]  # Store initial positions

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, num_particles)
        r2 = np.random.uniform(0, 1, num_particles)
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (
                swarm_best_position - particles)

        # Update positions
        particles += velocities

        # Store the positions for visualization
        particle_trajectories.append(np.copy(particles))

        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    # Return the best solution found by the PSO algorithm and trajectories
    return swarm_best_position, swarm_best_fitness, particle_trajectories

# Define the dimensions of the problem (1D)
dim = 1

# Run the PSO algorithm on the 1D Rastrigin function
solution, fitness, particle_trajectories = pso_1d_trajectories(rastrigin, dim=dim)

# Print the solution and fitness value
print('Solution:', solution)
print('Fitness:', fitness)

# Create a plot for visualization
x = np.linspace(-5.12, 5.12, 100)
y = rastrigin(x)

# Plot the Rastrigin function
plt.plot(x, y, label='Rastrigin Function')

# Assign random colors for each particle
colors = np.random.rand(len(particle_trajectories[0]), 3)

# Plot the trajectories of all particles
for i in range(len(particle_trajectories[0])):
    trajectory = [pos[i] for pos in particle_trajectories]
    plt.plot(trajectory, rastrigin(np.array(trajectory)), color=colors[i], linestyle='--', marker='o', label=f'Particle {i+1}')

# Highlight the best solution found
plt.scatter(solution, fitness, color='red', s=100, label='Best Solution', zorder=5)
plt.axvline(solution, color='red', linestyle='--', label='Solution Location')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('1D Rastrigin Function and PSO Particle Trajectories')
plt.legend()
plt.show()
