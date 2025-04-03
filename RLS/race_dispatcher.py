import numpy as np
import ray
import uuid
import time
import copy

from RLS.util.pool import Race

class RaceDispatcher:

    def init_race(self, instance_conf_assigment, instance_partition_id, cpu_size):
        """
        Create a new tournament out of the given configurations and list of instances.
        :param results: Results cache.
        :param configurations: List. Configurations for the tournament
        :param instance_partition: List. List of instances
        :param instance_partition_id: Id of the instance set.
        :return: Tournament, first conf/instance assignment to run
        """
        initial_instance_conf_assignments = instance_conf_assigment[:cpu_size]
        order = [[c_i[0].id, c_i[1]] for c_i in instance_conf_assigment]
        return Race(uuid.uuid4(), [], instance_conf_assigment,order, {}, instance_partition_id), initial_instance_conf_assignments




    def update_race(self, finished_conf, race):
        """
        Given a finishing conf we update the tournament if necessary. I.e the finishing conf has seen all instances of
        the tournament. In that case, it is moved either to the best or worst finishers. best finishers are ordered.
        Worst finishers are not
        :param results: Ray cache object.
        :param finished_conf: Configuration that finished or was canceled
        :param tournament: Tournament the finish conf was a member of
        :param number_winner: Int that determines the number of winners per tournament
        :return: updated tournament, stopping signal
        """

        race.c_i_pairs_nr.remove(finished_conf)
        race.c_i_pairs_d.append(finished_conf)


        # If there are no configurations left we end the tournament
        if len(race.c_i_pairs_nr) == 0:
            stop = True
        else:
            stop = False

        return race, stop

    def next_race_run(self, race):
        """
        Decided which conf/instance pair to run next. Rule: If the configuration that has just finished was not killed
        nor saw all instances, it is assigned a new instance at random. Else, the configuration with the lowest runtime
        so far is selected.
        :param results: Ray cache
        :param tournament: The tournament we opt to create a new task for
        :param finished_conf: Configuration that just finished before
        :return: configuration, instance pair to run next
        """
        possible_c_i = []
        for conf_i in race.c_i_pairs_nr:
            if conf_i[0].id not in race.ray_object_store.keys():
                possible_c_i.append(conf_i)

        if len(possible_c_i) == 0:
            return [[None, None]]
        else:
            return [possible_c_i[0]]




