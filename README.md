# AI Resource Allocation Solver

An intelligent agent designed to solve complex resource allocation problems. This tool assigns tasks to workers to minimize total costs while adhering to strict capacity constraints. 
It features custom implementations of state-space search algorithms without relying on external "black box" solvers.

---

## Key Features
* Algorithms: Implements A Search*, Uniform Cost Search (Dijkstra), and Depth-First Search (DFS) from scratch.
* Custom Heuristics: Uses a consistent & admissible heuristic (Problem Relaxation/Sum of Minimums) to guarantee optimal solutions with minimal node expansions.
* Constraint Satisfaction: Strictly enforces logic constraints (e.g., max_tasks_per_worker) during state expansion.
* Efficiency: Optimized for performance using heapq for priority management and NumPy for vectorized cost calculations.

## Available Algorithms:

* Astar-with-heuristic (A*)
* Astar-no-heuristic (Uniform Cost Search)
* depth-first 

## Input/Output
Input is a JSON file defining worker constraints and cost matrices - explore the 'testproblem'-folder to see their structure.
The tool outputs a solution.json containing the optimal assignment, step-by-step action sequence, and performance metrics (total cost, expansion count).

To test the programm, copy one of the testproblem .json files into the solutions.py directory.
Then run: 
  python solutios.py *algorithm_name* *testproblem.json*

