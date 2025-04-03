"""This module contains the point generation class."""

import uuid
import random
from RLS.util.generators.random_point_generator import random_point


class PointGen:
    """Interface for point generation."""

    def __init__(self, scenario, gm=random_point, seed=False):
        """
        Initialize PointGen.

        : param scenario: scenario object
        : param gm: point generating method to use
        """
        self.s = scenario
        self.gen_method = gm
        self.seed = seed

    def point_generator(self, **kwargs):
        """
        Running point generation according to object setting.

        : param meta: meta data a point generator requires
        : **kwargs: depend on gen_method
        return: configuration/point generated
        """
        if self.seed:
            self.id = uuid.UUID(int=random.getrandbits(self.seed))
        else:
            self.id = uuid.uuid4()
        configuration = self.gen_method(self.s, self.id, **kwargs)

        return configuration
