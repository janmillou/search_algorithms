# Imports:
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any
import os
import json
import argparse
import heapq
import re

class Heuristic:
    def __init__(self, specification):
         pass

    def estimate(self, state):
        return 0.

class Solver:
    def __init__(self, problem, heuristic):
        self.problem = problem
        self.heuristic = heuristic
    
    def get_solution(self, node):
        """
        Backtrack a solution.
        """
        raise NotImplementedError()

    def solve(self):
        """
        Try 10 random assignments and abort if no solution was found.
        """
        for i in range(10):
            # Note that the following random state could be infeasible
            random_state = tuple(np.random.randint(low=0, high=self.problem.num_workers, size=(self.problem.num_tasks,)))

            if self.problem.is_goal_state(random_state):
                return self.get_solution(None)
                        
        # If no solution is found
        raise ValueError("I was not able to solve the problem, sorry.")


#Problem Class (from task desciption):
class Problem:
    def __init__(self, specification:Dict[str, Any]):
        """
        Initializes the problem. Specification should be the content of the json-file with the problem description.
        """
        self.setup(specification)

    def setup(self, specification:Dict[str, Any]):
        """
        Parse problem configuration and store internally.
        """
        # store general problem information
        self.num_workers = specification["parameters"]["num_workers"]
        self.workers = list(range(self.num_workers))
        self.num_tasks = specification["parameters"]["num_tasks"]
        self.max_tasks_per_worker = specification["parameters"]["max_tasks_per_worker"]

        # parse costs for solving tasks with workers
        all_costs = []
        for task in specification["tasks"]:
            costs = np.zeros((self.num_workers,))
            for worker_str, cost in task["costs"].items():
                tokens = re.match("worker_([0-9]+)", worker_str)
                if tokens is not None:
                    wid = int(tokens.group(1))
                else:
                    raise ValueError(f"Can't parser {worker_str}.")
                costs[wid] = float(cost)
            all_costs.append(costs)
        self.all_costs = np.array(all_costs)

    def get_initial_state(self) -> Tuple[int, ...]:
        return self._encode(-np.ones((self.num_tasks,), dtype=np.int_))

    def _encode(self, state: NDArray[np.int_]) -> Tuple[int, ...]:
        """
        Converts a state given as an numpy array into a (hashable tuple).
        """
        return tuple(state)

    def _decode(self, state: Tuple[int, ...]) -> NDArray[np.int_]:
        """
        Converts a state given as a (hashable) tuple into a numpy ndarray.
        """
        return np.array(state, dtype=np.int_)
    
    def is_valid(self, state: NDArray[np.int_]):
        """
        Checks whether state is valid. Throws an exception if it is not valid.
        """
        assert isinstance(state, np.ndarray)
        assert len(state) == self.num_tasks
        assert np.all(state >= -1)
        assert np.all(state < self.num_workers)

        # check constraints on maximum tasks per worker
        tasks_per_worker = np.zeros((self.num_workers,), dtype=np.int_)
        for task_id, worker_id in enumerate(state):
            if worker_id >= 0:
                tasks_per_worker[worker_id] += 1
        assert np.all(tasks_per_worker <= self.max_tasks_per_worker)
    
    def is_goal_state(self, encoded_state: Tuple[int, ...]) -> bool:
        """
        Test whether state is a goal of the problem.
        The state must be provide as a tuple as this function is to be used by the problem solver.
        """
        state = self._decode(encoded_state)
        self.is_valid(state)
        try:
            if np.all([x in self.workers for x in state]):
                return True
            else:
                return False
        except:
            raise ValueError("Invalid state in goal test: ", state)
        
    def _open_tasks(self, state: NDArray[np.int_]) -> NDArray[np.int_]:
        unsolved_tasks = np.where(state == -1)[0]

        return unsolved_tasks
        
    def get_actions(self, encoded_state: Tuple[int, ...]):
        """
        Get list of possibel actions in a state.
        """
        state = self._decode(encoded_state)
        self.is_valid(state)

        # list of unsolved tasks
        unsolved_tasks = self._open_tasks(state)

        # any of the unsolved tasks can be assigned to any worker with free capacity
        tasks_per_worker = np.zeros((self.num_workers,), dtype=np.int32)
        for task_id, worker_id in enumerate(state):
            if worker_id >= 0:
                tasks_per_worker[worker_id] += 1
        workers_with_free_capacity = np.where(tasks_per_worker < self.max_tasks_per_worker)[0]
        actions = [(worker_id, task_id) for worker_id in workers_with_free_capacity for task_id in unsolved_tasks]
        return actions

    def state_cost(self, state: NDArray[np.int_]) -> float:
        """
        Compute the cost for the given state.
        """
        total_cost = np.sum([self.all_costs[task_id, worker_id] for worker_id, worker in enumerate(state) for task_id in worker])

        return total_cost

    def take_action(self, encoded_state: Tuple[int, ...], action: Tuple[int, int]) -> Tuple[Tuple[int, ...], float]:
        """ 
        Return state and cost for taking the given action in the given state.
        """
        state = self._decode(encoded_state)
        self.is_valid(state)
        assert isinstance(action, tuple)
        assert len(action) == 2
        
        worker_id, task_id = action
        unsolved_tasks = self._open_tasks(state)

        new_state = np.copy(state)
        new_state[action[1]] = action[0]       

        cost = self.all_costs[task_id, worker_id]

        return self._encode(new_state), cost


def save_solution(solution_data):
    output = {
        "solution": solution_data["solution"],
        "action_sequence":solution_data["action_sequence"],
        "n_expansions": solution_data["n_expansions"],
        "total_cost": solution_data["total_cost"]
    }

    with open('solution.json', 'w') as f:
        json.dump(output, f, indent=4)


def solve_no_heuristic(problem):
    print('started...')
    initial_state = problem.get_initial_state()
    tie_breaker_counter = 0
    frontier = []
    # frontier-queue should hold (cost, tie_breaker_counter, current_state, history) 
    heapq.heappush(frontier, (0.0, tie_breaker_counter, initial_state, [])) 
    explored_set = set()
    n_expansions = 0

    while frontier:
        cost, _, current_state, history = heapq.heappop(frontier)
        
        if problem.is_goal_state(current_state):
            print(f'Found solution: Cost: {cost}, Expansions: {n_expansions}')

            solution = {}
            state_dec = problem._decode(current_state)

            for task_id, worker_id in enumerate(state_dec):
                worker_key = f'worker_{worker_id}'
                if worker_key not in solution:
                    solution[worker_key] = []
                solution[worker_key].append(int(task_id))

            actions = {}
            for idx, action, in enumerate(history):
                actions[f'step_{idx}'] = [int(action[0]), int(action[1])]
            
            solution_data = {}
            solution_data['solution'] = solution
            solution_data['action_sequence'] = actions
            solution_data['n_expansions'] = n_expansions
            solution_data['total_cost'] = cost

            save_solution(solution_data)
            return
        
        if current_state in explored_set:
            continue
        explored_set.add(current_state)

        n_expansions += 1

        for action in problem.get_actions(current_state):
            next_state, step_cost = problem.take_action(current_state, action)

            # find some way of only 
            if next_state not in explored_set:
                new_cost = cost + step_cost
                new_history = history + [action]

                tie_breaker_counter += 1
                heapq.heappush(frontier, (new_cost, tie_breaker_counter, next_state, new_history))
    
    raise ValueError("problem could not be solved (frontier empty)")


def parser():
    parser = argparse.ArgumentParser(prog="solutions.py", description="solve resource allocation problems", epilog='siu')

    parser.add_argument('algorithm', type=str, 
                        choices=['Astar-no-heuristic', 'Astar-with-heuristic', 'depth-first'], 
                        help='select alorithm: [Astar-no-heuristic, Astar-with-heuristic, depth-first]')
    parser.add_argument('file_name', type=str, 
                        help='File name of .json problem file')

    args = parser.parse_args()

    print(f'Selected Algorithm: {args.algorithm}')
    print(f'Selected file: {args.file_name}')

    return args


if __name__ == "__main__":
    args = parser()

    try:
        with open(args.file_name, 'r') as file:
            problem_spec = json.load(file)

        problem = Problem(problem_spec)

        if args.algorithm == 'Astar-no-heuristic':
            solve_no_heuristic(problem)
        elif args.algorithm == 'Astar-with-heuristic':
            pass
        else:
            pass

    except Exception as e:
        print(f"error: {e}")
        raise ValueError(e)
    