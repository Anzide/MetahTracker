**Metah Tracker**
This is a program aims to track and observe how various algorithms run on given landscapes (fitness functions). Particularly, it focus on detailed observation on population-based metaheuristic algorithms, such as Particle Swarm Optimization and Differential Evolution.

**Quick Demo**

1. Install requirements.

> pip install -r requirements.txt

2. Go to **demo.py**, and try to run different functions in **main()**.

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

