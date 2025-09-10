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


def solve_cnf(file_path):
    # Load clauses from the CNF file
    clauses = load_cnf_file(file_path)

    # Initialize the Glucose solver
    solver = Glucose4()

    # Add clauses to the solver
    for clause in clauses:
        solver.add_clause(clause)

    start_time = time.time()

    # Solve the SAT problem
    solver.solve()

    end_time = time.time()

    duration = end_time-start_time

    solver.delete()

    return duration


# Path to your CNF file
file_path = '../input/ta/glucose_sat/uf200-01.cnf'

# Solve the CNF problem
solve_cnf(file_path)
