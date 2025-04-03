"""This module contains functions for random point generation."""

import numpy as np
import math
import sys
from RLS.util.pool import Configuration, ParamType, Generator
from RLS.util.generators.default_point_generator import (
    check_conditionals,
    check_no_goods
)


def reset_no_goods(s, config_setting):
    """
    Checking for no goods and resetting parameter values if violated.

    : param s: scenario object
    : param config_setting: parameter configuration
    return: config_estting adjusted to no goods
    """
    for ng in s.no_goods:

        violation = True

        if violation:
            ng_values = list(ng.values())
            config_set_values = []

            for ng_element in ng:
                config_set_values.append(config_setting[ng_element])

            if config_set_values == ng_values:
                configs_to_reset = []
                violation = True

                for params in s.parameter:
                    if params.name in ng:
                        configs_to_reset.append(params)

                new_setting = random_set_conf(configs_to_reset)

                for ns in new_setting:
                    config_setting[ns] = new_setting[ns]

            else:
                violation = False
        else:
            continue

    return config_setting


def reset_conditionals(s, config_setting, cond_vio):
    """
    Checking for conditionals and resetting parameter values if violated.

    : param s: scenario object
    : param config_setting: parameter configuration
    : param cond_vio: list of parameters that violate conditionals
    return: config_estting adjusted to no goods
    """
    for cv in cond_vio:
        for ps in s.parameter:
            if cv == ps.name:
                # if ps.type == ParamType.categorical:
                config_setting[cv] = ps.default

    return config_setting


def random_set_conf(parameter):
    """
    Generating random configuration values for given param space.

    : param parameter: dataclass Parameter, filled out with scenario data
    return: randomly set parameters
    """
    config_setting = {}

    for param in parameter:

        if param.type == ParamType.categorical:
            pv = np.random.choice(param.bound)
            if isinstance(pv, np.bool_):
                config_setting[param.name] = bool(pv)
            else:
                config_setting[param.name] = pv

        elif param.type == ParamType.continuous:
            if param.scale:
                # Generate in logarithmic space
                if param.bound[1] < 0:
                    # Upper and lower bound are negative
                    config_setting[param.name] \
                        = -1 * math.exp(np.random.uniform(
                            low=math.log(param.bound[0] * -1),
                            high=math.log(param.bound[1] * -1)))

                elif param.bound[0] < 0:
                    # Lower bound is negative
                    lb = -1 * math.exp(np.random.uniform(
                        low=math.log(sys.float_info.min),
                        high=math.log(param.bound[0] * -1)))
                    ub = math.exp(np.random.uniform(
                        low=math.log(sys.float_info.min),
                        high=math.log(param.bound[1])))
                    config_setting[param.name] \
                        = np.random.choice([lb, ub])

                else:
                    # Bounds are positive
                    config_setting[param.name] \
                        = math.exp(np.random.uniform(
                            low=math.log(param.bound[0]),
                            high=math.log(param.bound[1])))
            else:
                config_setting[param.name] \
                    = np.random.uniform(low=param.bound[0],
                                        high=param.bound[1])

        elif param.type == ParamType.integer:
            if param.scale:
                # Generate in logarithmic space
                if param.bound[1] < 0:
                    # Upper and lower bound are negative
                    config_setting[param.name] \
                        = int(-1 * math.exp(np.random.uniform(
                            low=math.log(param.bound[0] * -1),
                            high=math.log(param.bound[1] * -1))))

                elif param.bound[0] < 0:
                    # Lower bound is negative
                    lb = int(-1 * math.exp(np.random.uniform(
                        low=math.log(sys.float_info.min),
                        high=math.log(param.bound[0] * -1))))
                    ub = int(math.exp(np.random.uniform(
                        low=math.log(sys.float_info.min),
                        high=math.log(param.bound[1]))))
                    config_setting[param.name] \
                        = np.random.choice([lb, ub])

                else:
                    # Bounds are positive
                    config_setting[param.name] \
                        = int(math.exp(np.random.uniform(
                            low=math.log(param.bound[0]),
                            high=math.log(param.bound[1]))))
            else:
                config_setting[param.name] \
                    = np.random.randint(low=param.bound[0],
                                        high=param.bound[1])

    return config_setting


def random_point(s, identity, seed=False):
    """
    Random parameter setting is generated in Configuration format.

    : param s: scenario object
    : param identity: uuid to identify configuration
    return: randomly set configuration, which accounts for no goods
    and conditionals
    """
    if seed:
        np.random.seed(seed)

    # Generate configuration randomly based on given parameter space
    config_setting = random_set_conf(s.parameter)

    '''
    # Check conditionals and turn off parameters if violated
    cond_vio = check_conditionals(s, config_setting)
    if cond_vio:
        config_setting = reset_conditionals(s, config_setting, cond_vio)
    '''

    # Check no goods and reset values if violated
    ng_vio = check_no_goods(s, config_setting)
    while ng_vio:
        config_setting = reset_no_goods(s, config_setting)
        ng_vio = check_no_goods(s, config_setting)

    # Fill Configuration class with ID and parameter values
    configuration = Configuration(identity, config_setting, Generator.random)

    return configuration
