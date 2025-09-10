import re
import warnings
from RLS.util.pool import Parameter, ParamType

import numpy as np

boolean_yes = ["on", "yes", "true"]
boolean_no = ["no", "off", "false"]
boolean_options = boolean_no + boolean_yes

def get_ta_arguments_from_pcs(para_file):
    """
     Read in a file that contains the target algorithm parameters. The file is .pcs and adheres to the structure of
     AClib.
    :param para_file: str. Path to the .pcs file
    :return: list, list. A list containing information on the parameters of the target algorithm & a list containing
    parameter value assignments that are not possible
    """

    no_goods = []
    parameters = []
    conditionals = {}

    with open(para_file, 'r') as pf:
        for line in pf:
            line = line.strip().split("#", 1)[0]

            # skip empty lines
            if line == "":
                continue

            try:
                line_split = line.split(None, 1)
                param_name = line_split[0].replace(" ", "")
                param_info = line_split[1]
            except IndexError:
                warnings.warn(f"Ignoring malformed or incomplete line in .pcs: '{line}'")
                continue


            if "|" not in param_info:

                # cat
                if '{' in param_info:
                    param_type, bounds, defaults, original_bound = get_categorical(param_name, param_info)
                    if param_type != None:
                        parameters.append(Parameter(param_name, param_type, bounds, defaults, {}, '', original_bound))
                # forbidden
                elif '{' in param_name:
                    no_good = get_no_goods(line, parameters)
                    no_goods.append(no_good)
                # cont.
                elif '[' in param_info:
                    param_type, bounds, defaults, scale = get_continuous(param_name, param_info)
                    parameters.append(Parameter(param_name, param_type, bounds, defaults, {}, scale, []))

            # conditionals
            elif '|' in param_info:
                condition_param, condition = get_conditional(param_name, param_info, parameters)

                if param_name not in conditionals:
                    conditionals[param_name] = {condition_param: condition}
                else:
                    conditionals[param_name].update({condition_param: condition})

            else:
                raise ValueError(f"The parameter file {para_file} contains unreadable elements. Check that"
                                 f" the structure adheres to AClib")

    # adding conditionals to parameters
    for pc in conditionals:
        condition_found = False
        for parameter in parameters:
            if re.search(r'\b' + str(pc) + r'\b', parameter.name):
                parameter.condition.update(conditionals[pc])
                condition_found = True
        # This should only be a warning: We may have conditions for cat. parameters that are not configurable.
        # We ignore these.
        if not condition_found:
            warnings.warn(f"Condition {pc}|{conditionals[pc]} will be dropped since either {pc} is "
                          f"not configurable or does not exist")

    return parameters, no_goods, conditionals


def get_categorical(param_name, param_info):
    """
    For a categorical parameter: check if its parsed attributes are valid and extract information on the parameter
    :param param_name: name of the parameter
    :param param_info: raw parameter information
    :return: param_type , bounds, defaults of the parameter
    """

    bounds = re.search(r'\{(.*)\}', param_info).group().strip("{ }").split(",")
    bounds = [b.replace(" ","") for b in bounds]
    defaults = re.findall(r'\[(.*)\]*]', param_info)
    original_bound = []

    # When len(bounds) ==1: The parameter can not be configured. Later we have to also raise a warning that conditions
                                                                # on it are ignored

    if bounds[0] in boolean_options:
        param_type = ParamType.categorical
        original_bound = bounds

        if defaults[0] in boolean_yes:
            defaults = True
        elif defaults[0] in boolean_no:
            defaults = False
        else:
            raise ValueError(f"For parameter {param_name} the parsed defaults are not within [yes, no, on, off]")
        bounds = [b in boolean_yes for b in bounds]
        bounds = sorted(bounds)

    elif isinstance(str(bounds[0]), str) & isinstance(str(defaults[0]), str):
        param_type = ParamType.categorical
        defaults = str(defaults[0])
        bounds = [str(b).replace(" ", "") for b in bounds]
        if defaults not in bounds:
            raise ValueError(f"For parameter {param_name} the default value is not within the range of the bounds")

    else:
        raise ValueError(f"For parameter {param_name} the parsed bounds were not boolean or categorical")

    return param_type, bounds, defaults, original_bound


def get_continuous(param_name, param_info):
    """
    For a continuous parameter: check if its parsed attributes are valid and extract information on the parameter
    :param param_name: name of the parameter
    :param param_info: raw parameter information
    :return: param_type , bounds, defaults,scale of the parameter
    """

    scale = re.search(r'[a-zA-Z]+', param_info)
    param_info = re.findall(r'\[[^\]]*]', param_info)
    bounds = param_info[0].strip("[] ").split(",")
    defaults = param_info[1].strip("[] ")

    # checking for set scale
    if scale and "i" in scale.group():
        param_type = ParamType.integer
        scale = scale.group().strip("i")

        if isinstance(int(defaults), int):
            defaults = int(defaults)
        else:
            raise ValueError(f"For parameter {param_name} the parsed defaults are not integer")
        bounds = [int(b) for b in bounds]

    else:
        param_type = ParamType.continuous

        if isinstance(float(defaults), float):
            defaults = float(defaults)
        else:
            raise ValueError(f"For parameter {param_name} the parsed defaults are not continuous")
        bounds = [float(b) for b in bounds]

        if scale is None:
            scale = ''
        else:
            scale = scale.group()

    if not bounds[0] <= defaults <= bounds[1]:
        raise ValueError(f"For parameter {param_name} the default value is not within the range of the bounds")


    return param_type, bounds, defaults, scale

def get_conditional(param_name, param_info, parameters):
    """
    For a parameter: get the information on conditionals
    :param param_name: name of the parameter
    :param param_info: raw parameter information
    :return: condition_param, condition
    """
    param_info = param_info.strip(" | ")

    condition = re.search(r'\{(.*)\}', param_info).group().strip("{ }").split(",")
    condition_param = re.search(r'.+?(?= in)', param_info).group().replace(" ", "")

    for p in parameters:
        if p.name == condition_param:
            p_type = p.type

    if condition[0] in boolean_options:
        condition = [c in boolean_yes for c in condition]
    elif p_type == ParamType.categorical:
        condition = [str(c).strip(" ") for c in condition]
    elif p_type == ParamType.continuous:
        condition = [float(c) for c in condition]
    elif p_type == ParamType.integer:
        condition = [int(c) for c in condition]
    else:
        raise ValueError(f"For parameter {param_name} the parsed conditions could not be read")

    return condition_param, condition

def get_no_goods(no_good, parameters):
    """
    Takes an string of form: {param_1=value_1 , param_2=value_2, ...} and returns a dic of the no good
    :param no_good: str. Takes an string of form: {param_1=value_1 , param_2=value_2, ...}
    :return: dic
    """

    forbidden = {}
    no_good = no_good.strip("{ }").split(",")

    for ng in no_good:
        param, value = ng.split("=")
        param = param.strip()
        value = value.strip()

        for p in parameters:
            if p.name == param:
                p_type = p.type

        if value in boolean_yes:
            value = True
        elif value in boolean_no:
            value = False
        elif p_type == ParamType.continuous:
            value = float(value)
        elif p_type == ParamType.integer:
            value = int(value)
        elif p_type == ParamType.categorical:
            value = str(value)
        else:
            raise ValueError(f"For no good {no_good} the parameter values are not known")

        forbidden[param] = value

    return forbidden


def read_instance_paths(instance_set_path):
    """
    Read in instances from an AClib instance file
    :param instance_set_path: str. Path to the instance file
    :return: list. List of paths to the instances
    """

    instance_set = []

    with open(instance_set_path, 'r') as f:
        for line in f:
            instance_set.append(line.strip())

    seen = set()
    uniq = []
    for i in instance_set:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
        else:
            warnings.warn(f"Instance {i} is not unique in the train set")

    return instance_set

def read_instance_features(feature_set_path):
    """
    Read in features from an AClib features file
    :param feature_set_path: str. Path to the feature file
    :return: dic, list. Dic with the read in features list with the feature names
    """

    features = {}
    with open(feature_set_path, 'r') as f:
        lines = f.readlines()
        feature_names = lines[0].strip().split(",")[1:]

        for line in lines[1:]:
            line = line.strip().split(",")
            if line[0] != "":
                features[line[0]] = np.array(line[1:], dtype=np.single)
    return features, feature_names
