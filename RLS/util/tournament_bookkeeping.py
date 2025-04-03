import os
from RLS.util.ta_execution import tae_from_cmd_wrapper_rt, tae_from_cmd_wrapper_quality
import time
import logging



def get_race_membership(races, conf):
    """
    For a list of tournaments, determine of which a conf is a member.
    :param tournaments: List
    :param conf: Conf
    :return:
    """
    for r in races:
        configuration_ids = [c_i[0] for c_i in r.c_i_order]
        if conf.id in configuration_ids:
            return r

def get_get_tournament_membership_with_ray_id(task_id, tournaments):
    """
    For a ray task id return the tournament it belongs to
    :param task_id:
    :param tournaments:
    :return:
    """

    ob_t = None
    for t in tournaments:
        t_objects = t.ray_object_store
        for confs, instance_objects in t_objects.items():
            for inst, ob in instance_objects.items():
                if ob == task_id:
                    ob_t = t
                    pass
    return ob_t

def get_tasks(taskdic, tasks):
    """
    Map back a ray object to the conf/instance pair.
    :param taskdic: Nested dic of {conf: {instance: ray object}}
    :param tasks: List with ray objects that are running
    :return: List of [conf, instances] pairs that are currently running
    """
    running_tasks = []
    for conf, instance in taskdic.items():
        for instance_name, object in instance.items():
            if object in tasks:
                running_tasks.append([conf, instance_name])
    return running_tasks

def update_tasks(tasks, next_task, tournament, global_cache, ta_wrapper, scenario):
    """

    :param tasks: List of ray objects
    :param next_task: List of [conf, instance] pairs
    :param tournament: Tournament the next task is part of
    :param global_cache: Ray cache
    :param ta_wrapper:
    :param scenario:
    :return: Updated list of ray objects
    """
    for t in next_task:
        if t[1] is not None:
            # TODO need to change the wrapper to something more generic here
            if scenario.run_obj == "runtime":
                task = tae_from_cmd_wrapper_rt.remote(t[0], t[1], global_cache, ta_wrapper, scenario)
            elif scenario.run_obj == "quality":
                task = tae_from_cmd_wrapper_quality.remote(t[0], t[1], global_cache, ta_wrapper, scenario)
            tasks.append(task)
            # We also add the ray object id to the tournament to latter map the id back
            if t[0].id not in tournament.ray_object_store.keys():
                tournament.ray_object_store[t[0].id] = {t[1]: task}
            else:
                tournament.ray_object_store[t[0].id][t[1]] = task
    return tasks


def termination_check(termination_criterion, main_loop_start, total_runtime, total_tournament_number,
                      tournament_counter):
    """
    Check what termination criterion for the main tournament loop has been parsed and return true,
    if the criterion is not met yet.
    :param termination_criterion: Str. termination criterion for the tournament main loop
    :param main_loop_start: Int. Time of the start of the tournament main loop
    :param total_runtime: Int. Total runtime for the main loop, when the termination criterion is "total_runtime"
    :param total_tournament_number: Int. Total number of tournaments for the main loop,
                                    when the termination criterion is "total_tournament_number"
    :param tournament_counter: Int. Number of tournaments, that finished already
    :return: Bool. True, when the termination criterion is not met, False otherwise
    """
    if termination_criterion == "total_runtime":
        return time.time() - main_loop_start < total_runtime

    elif termination_criterion == "total_tournament_number":
        return tournament_counter < total_tournament_number

    else:
        return time.time() - main_loop_start < total_runtime



