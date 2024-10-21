**Metah Tracker**

This is a project aims to track and observe how various algorithms run on given landscapes (fitness functions). Particularly, it focus on detailed observation on population-based metaheuristic algorithms, such as Particle Swarm Optimization and Differential Evolution (work-in-progress).

**Quick Demo**

1. Install requirements.

> pip install -r requirements.txt

2. Run **demo.py**, and your will see two dynamic plots (the second appears after closing the first one). The first plot shows a two-dimension landscape where fitness function is the Rastrigin function, and the second shows the process of Particle Swarm Optimization algorithm being executed on the landscape.

<img src="https://i.ibb.co/W0P6rTT/img0.png" style="zoom: 50%;" /><img src="https://i.ibb.co/vXtLwC5/img1.png" style="zoom: 60%;" />

3. You could try to use a different landscape in **main()** by commenting choose_rastrigin_landscape() and uncommenting another:

```
# ls = choose_decaying_cosine_landscape()
ls = choose_random_parabola_landscape()
# ls = choose_rastrigin_landscape()
```

Currently there is only one algorithm implemented, which is PSO, but you could add your owns by imitating it. Differential Evaluation algorithm is on the to-do list.

**Design Your Own Experiment**

First, introduce a few key concepts. All words highlighted in bold are actual existing classes. **DimSeparatedLandscape** describes a landscape that is separated by dimensions. That is to say, each dimension has its own function, and the fitness of a point is the sum of the fitness of each dimension. Take a 3-dim **DimSeparatedLandscape** as example, its fitness function can be expressed as following:  
$$
F(x, y, z) = f_1(x)+f_2(y)+f_3(z)
$$
The function on each dimension is a **UnaryFunction**, which has an additional condition that it must be analyzable for local optimum. Local optimum analyzability means that given an arbitrary point, the coordinate of corresponding local optima could be calculated. Using the one-dim Rastrigin function as an example, the way to calculate the local optimum (which is local minimum in this case) to given point is rounding the given coordinate to the nearest integers. If you wish to create your own **UnaryFunction**, you will have to implement the local_optimum method in it. The implementation of the existing functions can be referred to.

With the **UnaryFunction** in place (existing or self-created), we can design the experiment. 

```python
# Consturct landscape by assigning the number of dimensions, OptimizationType, UnaryFunction on each dimension and boundary on each dimension
landscape = DimSeparatedLandscape(2,
                   OptimizationType.MINIMISATION,
                   [RastriginFunc(), RastriginFunc()],
                   [(-5.12, 5.12), (-5.12, 5.12)])
# Assign landscape and parameters for algorithm
pso = PsoAlgorithm(ls, dim=2, particle_num=40, max_iter=20, seed=123)
# Run
pso.run()
# Show the algorithm process, only works if the landscpae is 2D. If it's not 2D, you could collect and store the process information by editing the algorithm. 
ls.plot_2d_exploration(frame_interval=1000, results=pso.results)
```

You might want to create new implementation of **UnaryFunction** and **Algorithm** for creating and testing different landscapes and algorithms that not yet existing.

**Noteworthy Classes and Functions**

class **Landscape**:

- plot_1d_func: Plot an 1d landscape.
- plot_2d_exploration: Plot the exploration process of an algorithm on a two-dim landscape. This function is the only plotting function that has the ability to animate the exploration process.
- plot_2d_func_surface: Plot a 2D landscape in a 3D surface style.

class **DimSeparatedLandscape**:

- \_\_init\_\_: Used to construct a DimSeparatedLandscape.
- compare_solution: Compare two solutions and return comparison: Is it an exploration or exploitation? Accepted or Rejected? Successful or Failed?

Others:

- Function **auto_select_mpl_backend** in **basic.utils**: Select the appropriate backend for matplotlib to show the plots. Only being tested on Windows.

- Classes **OptimizationType** and **MoveType** in **enums.py**: They jointly define the evaluation criteria for any update movement of an individual.
- Class **Algorithm**.

**Known Issues**

- The definition of MoveType might not be completely correct for PSO, cause there is not actual rejection in the current implementation of PSO - any vector update is applied to the position of particle, with no rejecting mechanism.  
- Any other known issues could be found by searching comment "TODO" in the whole project.

**Acknowledgments**

I would like to express my sincere gratitude to Professor [Stephen Chen](https://profiles.laps.yorku.ca/profiles/sychen/) for giving me the opportunity to undertake the internship, as well as for his invaluable supervision and advice throughout my research. I am also deeply thankful to Professor [Tim Hendtlass](https://www.researchgate.net/profile/Tim-Hendtlass) for his insightful suggestions and the chance to refer his previous work, without which this project might not have been completed.

