

import sys
import os
sys.path.append(os.getcwd())

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import importlib
import logging
import numpy as np
import ray
import pickle
import gzip

from args_parse import parse_args
from RLS.util.scenario import Scenario
from RLS.util.log_setup import check_log_folder
from search_loop import offline_rl_search_loop


sys.path.append(os.getcwd())


if __name__ == "__main__":
    rl_args = parse_args()
    # print(rl_args)


    # make the wrapper that gives the ta commands a module
    wrapper_mod = importlib.import_module(rl_args["wrapper_mod_name"])
    wrapper_name = rl_args["wrapper_class_name"]
    wrapper_ = getattr(wrapper_mod, wrapper_name)
    ta_wrapper = wrapper_()

    scenario = Scenario(rl_args["scenario_file"], rl_args)
    print(scenario.run_obj)

    # we may need to restrict the number of cpus
    if scenario.num_cpu is None:
        scenario.num_cpu = scenario.racesize

    check_log_folder(scenario.log_folder)

    logging.\
        basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler(
                        f"{scenario.log_folder}/main.log"), ])

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {scenario.log_folder}")
    # init
    if scenario.localmode:
        ray.init()
    else:
        ray.init(address="auto")

    offline_rl_search_loop(scenario, ta_wrapper, logger)

    ray.shutdown()
