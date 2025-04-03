from dataclasses import dataclass
from enum import Enum


@dataclass
class Configuration:
    id: int
    conf: dict
    generator: Enum

@dataclass
class Parameter:
    name: str
    type: str
    bound: list
    default: int
    condition: list
    scale: str
    original_bound: list

@dataclass
class Race:
    id: int
    c_i_pairs_d: list
    c_i_pairs_nr: list
    c_i_order: list
    ray_object_store: dict
    instance_set_id: int

class ParamType(Enum):
    categorical = 1
    continuous = 2
    integer = 3


class Generator(Enum):
    default = 1
    random = 2
    var_graph = 3
    lhc = 4
    smac = 5
    ggapp = 6
    cppl = 7

