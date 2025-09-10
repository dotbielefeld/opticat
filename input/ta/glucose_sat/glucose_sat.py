import sys
import numpy as np
from pysat.solvers import Glucose4
import time

def load_cnf_file(file_path):

    """Load CNF clauses from a file."""

    clauses = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c') or line.startswith('p'):
                continue

            clause = [int(x) for x in line.strip().split() if x != '0']

            if clause:
                clauses.append(clause)

    return clauses


def solve_cnf(file_path, parameters):
    # Load clauses from the CNF file
    clauses = load_cnf_file(file_path)

    # Initialize the Glucose solver
    solver = Glucose4(**parameters)

    # Add clauses to the solver
    for clause in clauses:
        solver.add_clause(clause)

    start_time = time.time()

    # Solve the SAT problem
    solver.solve()

    end_time = time.time()

    # Calculate run time and return value
    duration = end_time-start_time

    solver.delete()

    #print(f"Duration: {duration}")

    return duration


def parse_args():
    """Parses command-line arguments passed from `command_generator.py`."""
    args = sys.argv[1:]  # Ignore the script name
    config = {}

    for arg in args:
        if arg.startswith("--"):
            key, value = arg[2:].split("=")
            try:
                config[key] = float(value)  # Convert to float if possible
            except ValueError:
                config[key] = value  # Keep as string if not convertible

    return config


if __name__ == "__main__":
    config = parse_args()
    
    # Extract parameters
    parameters = [value for key, value in sorted(config.items())]# if key.startswith("x")]

    if len(parameters) == 0:
        print("Error: No valid parameters provided.")
        sys.exit(1)

    file_path = 'uf200-01.cnf'

    print("Running solve_cnf now")
    result = solve_cnf(file_path, parameters)
    print(f"output: {result}")
    time.sleep(0.3)
    #cpu_intensive_wait(20)
