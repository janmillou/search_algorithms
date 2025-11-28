# Imports:
import numpy as np
import os
import json
import argparse
import heapq
import re



# Eingelesenes Problem mittels Suche lösen & Details(Kosten, etc) ausgeben
# json datei erzeugen

#def no_heuristic():

    
# function that parses command line arguments:
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