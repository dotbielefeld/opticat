import numpy as np
import math
import numpy as np
from scipy.spatial import distance



class InstanceSetRACE:
    def __init__(self, instance_set, start_instance_size, set_size=None, instance_increment_size= None, target_reach=None):
        """

        :param instance_set: set of instances available.
        :param start_instance_size: size of the first instances se to be created
        :param set_size: If not set the biggest instance set to be created includes all instances in instance_set. If
        set the biggest instance set will be of the size of the given int.
        """
        self.instance_set = instance_set
        self.start_instance_size = start_instance_size
        self.number_to_sample = start_instance_size
        self.instance_sets = []
        self.subset_counter = 0

        if set_size and set_size <= len(instance_set):
            self.set_size = set_size
        else:
            self.set_size = len(instance_set)

        #if instance_increment_size :
        self.instance_increment_size = instance_increment_size
        #else:
       #     self.instance_increment_size = self.set_size / np.floor(self.set_size/self.start_instance_size)

        if target_reach and instance_increment_size > 0:
            set_size = self.start_instance_size
            counter = 0
            while set_size <= self.set_size :
                set_size += self.instance_increment_size
                counter += 1
            self.target_increment = np.ceil(target_reach / counter)
        else:
            self.target_increment = 1

    def get_subset(self, next_tournament_set_id):
        """
        Create an instance set for the next tournament. The set contains all instances that were included in the
        previous sets as well as a new subset of instances.
        :param next_tournament_set_id: int. Id of the subset to get the next instances for .
        :return: id of the instances set, list containing instances and previous instances of the subset
        """
        assert next_tournament_set_id <= self.subset_counter
        # If we have already created the required subset we return it

        if not self.instance_sets:
            if self.start_instance_size > len(self.instance_set):
                raise ValueError("The number of instances provided is smaller than the initial instance set size")
            else:
                new_subset = np.random.choice(self.instance_set, self.number_to_sample, replace=False).tolist()
                self.instance_sets.append(new_subset)
        else:
            if next_tournament_set_id % self.target_increment == 0:
                self.number_to_sample = int(min(self.number_to_sample + self.instance_increment_size, self.set_size))


            new_subset = np.random.choice(self.instance_set, self.number_to_sample, replace=False).tolist()
            self.instance_sets.append(new_subset)

        self.subset_counter += 1

        return next_tournament_set_id, new_subset