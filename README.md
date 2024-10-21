**Metah Tracker**

This is a program aims to track and observe how various algorithms run on given landscapes (fitness functions). Particularly, it focus on detailed observation on population-based metaheuristic algorithms, such as Particle Swarm Optimization and Differential Evolution.

**Quick Demo**

1. Install requirements.

> pip install -r requirements.txt

2. Run demo.py, and your will see two plots (the second appears after closing the first one).

![img0](../image/img0.png?raw=true)
![img1](../image/img1.png?raw=true)

2. Try to use different landscapes in **main()**:

```
# ls = choose_decaying_cosine_landscape()
# ls = choose_random_parabola_landscape()
# ls = choose_rastrigin_landscape()
```

// TODO: Add details 

**Experiment Execution**

There are two types of experiments you could conduct with this program.

a) Basic Experiment. 

In basic experiment, an **Evaluator** **(Fitness Function)** is used to construct a **Landscape**, and then run **Algorithm** on it.

b) Advanced Experiment. 

Advanced experiment is an extend of basic experiment, and aims on tracking the exploration process. A series of **Unary Function** is used to construct a **DimSeparatedLandscape**, and then run **Algorithm** on it.

**Concepts**

- **Evaluator**: equals to Fitness Function.
- **Landscape**: A landscape contains a function that maps a point to a fitness value. The function is called Evaluator.
- **Dim Separated Function**

**Configure your own algorithm and landscape**

