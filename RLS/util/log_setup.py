import os
import shutil
from datetime import datetime
import dataclasses
import json
import uuid
from enum import Enum
import ray
import numpy as np

def check_log_folder(folder_for_run = None):
    if folder_for_run == None:
        folder_for_run = "latest"

    ff = ""
    for f in folder_for_run.split("/"):
        ff = ff + "/" + f
        if not os.path.exists(f'./{ff}'):
            os.makedirs(f'./{ff}')


class TournamentEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            data_dic = dataclasses.asdict(o)
            if "ray_object_store" in data_dic.keys():
                del data_dic["ray_object_store"]
            return data_dic
        elif isinstance(o, uuid.UUID):
            return str(o)
        elif isinstance(o, Enum):
            return str(o)
        elif isinstance(o, ray._raylet.ObjectRef):
            return str(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, dict):
            for k in o.keys():
                if isinstance(k, uuid.UUID):
                    o[str(k)] = o.pop(k)
            return o

        return super().default(o)

class ConfEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super().default(obj)
