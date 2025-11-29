# Imports:
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any
import os
import json
import argparse
import heapq
import re



# Eingelesenes Problem mittels Suche lösen & Details(Kosten, etc) ausgeben
# json datei erzeugen

#def no_heuristic():
"""
Problem Class (from task desciption):
"""
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
    
"""
depth-first-search() & recursive_dfs():

def recusive_dfs():

def depth-first-search(problem):
    return recursive_dfs()
"""

"""
function that parses command line arguments:
"""
def parser(problem):
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

def main():
    args = parser()

    if args.algotithm == 'Astar-no-heuristic':
        # no_heuristic(args.file_name)


if __name__ == "__main__":
    main()