import copy

from args_parse import parse_args
from ppo_ac_beta import PPO_AC_BETA
from ppo_ac_normal import PPO_AC_NORMAL
from RLS.util.scenario import Scenario
import numpy as np
from RLS.instance_features import Feature_Transfomer
import os
import json
from RLS.util.log_setup import TournamentEncoder
from RLS.util.point_gen import PointGen
from RLS.util.generators.default_point_generator import default_point, check_conditionals
import torch
from ac_util import make_prediction


if __name__ == "__main__":
    rl_args = parse_args()

    scenario = Scenario(rl_args["scenario_file"], rl_args)

    if scenario.feature_free is False:
        scenario.features = {key: value for key, value in scenario.features.items()}

        feature_keys = list(scenario.features.keys())
        feature_values = list(scenario.features.values())
        fm = np.array(feature_values)

        scaler = Feature_Transfomer(delete_features=scenario.delete_features)
        scaler.load_transformer(f'{scenario.log_folder}/models/transformer.pkl')

        fm = scaler.transform(fm)
        for fk, fv in zip(feature_keys, fm):
            scenario.features[fk] = fv

        state_dim = fm.shape[1]
    else:
        state_dim = 10

    if scenario.rl == "beta":
        rl_agent = PPO_AC_BETA(scenario.parameter, state_dim, scenario)
    elif scenario.rl == "normal":
        rl_agent = PPO_AC_NORMAL(scenario.parameter, state_dim, scenario)


    rl_agent.actor = torch.load(f'{scenario.log_folder}/models/actor_final.pth')

    # Process training instances
    if scenario.instance_file:
        make_prediction(
            instances=scenario.instance_set,
            scenario=scenario,
            rl_agent=rl_agent,
            output_file=f"{scenario.log_folder}/final_training.json"
        )

    # Process test instances
    if scenario.test_instances:
        make_prediction(
            instances=scenario.test_instances,
            scenario=scenario,
            rl_agent=rl_agent,
            output_file=f"{scenario.log_folder}/final_test.json"
        )
