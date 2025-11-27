# Programmieraufgabe 1

# Imports:
import numpy as np
import os
import json
import argparse
import heapq
import re

# Kommandozeilenevent für 
# (Suchalgorithmus(Astar-no-heuristic, Astar-with-Heuristic, depth-first), 
# Dateiname)


# Eingelesenes Problem mittels Suche lösen & Details(Kosten, etc) ausgeben
# json datei erzeugen

def validate_alg(alg_name):
    valid_algs = ['Astar-no-heuristic', 'Astar-with-Heuristic', 'depth-first']
    if alg_name not in valid_algs:
        raise Exception('invalid algorithm')

def validate_file(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as problem:
            data = json.load(problem)
            print(json.dumps(data, indent=4))
    else:
        raise Exception("invalid filename")

def main():
    #alg_name = input("Choose Algorithm:")
    #validate_alg(alg_name)
    file_name = input("Enter filename ")
    validate_file(file_name)
    

if __name__ == "__main__":
    main()