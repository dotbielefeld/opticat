
import math
import copy


def get_mean_rt_over_instance(results, instances, timeout, par_penalty=1):
    sums = {key: 0 for key in instances}
    counts = {key: 0 for key in instances}
    squares = {key: 0 for key in instances}

    # Traverse the nested dictionary to populate sums and counts
    for outer_key, inner_dict in results.items():
        for key in instances:
            if key in inner_dict:
                rt = inner_dict[key]
                if math.isnan(rt):
                    rt = timeout * par_penalty
                sums[key] += rt
                counts[key] += 1
                squares[key] += rt**2

    means = {key: (sums[key] / counts[key]) if counts[key] > 0 else None for key in instances}
    stds = {key: (math.sqrt(squares[key] / counts[key] - (sums[key] / counts[key]) ** 2) if counts[key] > 0 else None) for key in instances}
    return means, stds

def get_mean_rt_for_batch(results, confs, instances, timeout, par_penalty=1):
    sums = {key: 0 for key in instances}
    counts = {key: 0 for key in instances}

    for conf in confs:
        conf_results = results[conf]
        for i in instances:
            if i in conf_results:
                rt = conf_results[i]
                if math.isnan(rt):
                    rt = timeout * par_penalty
                sums[i] += rt
                counts[i] += 1
    means = {key: (sums[key] / counts[key]) if counts[key] > 0 else None for key in instances}
    return means


def get_instances_no_results(results, configuration_id, instance_set):
    """
    For a configuration get a list of instances we have no results for yet
    :param results: Dic of results: {conf_id: {instance: runtime}}
    :param configuration_id: Id of the configuration
    :param instance_set: List of instances
    :return: List of configuration the conf has not been run on
    """
    not_run_on= copy.deepcopy(instance_set)

    if configuration_id in results.keys():
        configuration_results = results[configuration_id]

        instances_run_on = configuration_results.keys()

        for iro in instances_run_on:
            if iro in not_run_on:
                not_run_on.remove(iro)

    return not_run_on

