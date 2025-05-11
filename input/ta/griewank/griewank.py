import sys
import numpy as np
import time


def cpu_intensive_wait(duration):
    start_time = time.time()
    while (time.time() - start_time) < duration:
        sum(i*i for i in range(10000))  # Dummy computation


def griewank(x):
    """Computes the Griewank function for a given input vector x."""

    return 1 + sum([(xi ** 2)-600 for xi in x]) / 4000 - np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])


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
    
    # Extract parameter values
    x = [value for key, value in sorted(config.items()) if key.startswith("x")]

    if len(x) == 0:
        print("Error: No valid parameters provided.")
        sys.exit(1)

    result = griewank(x)
    print(f"output: {result}")
    time.sleep(0.3)
    #cpu_intensive_wait(20)
