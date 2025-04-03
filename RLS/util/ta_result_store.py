import ray
import logging
import json
from RLS.util.log_setup import TournamentEncoder



@ray.remote(num_cpus=1)
class TargetAlgorithmObserver:

    def __init__(self, scenario):
        self.intermediate_output = {}
        self.results = {}
        self.start_time = {}
        self.race_history = {}
        self.termination_history = {}
        self.races = {}
        self.scenario = scenario
        self.last_race = None

        logging.basicConfig(filename=f'{self.scenario.log_folder}/Target_Algorithm_Cache.logger', level=logging.INFO,
                            format='%(asctime)s %(message)s')


    def put_result(self, conf_id, instance_id, result):
        logging.info(f"Getting final result: {conf_id}, {instance_id}, {result}")
        if conf_id not in self.results:
            self.results[conf_id] = {}

        if instance_id not in self.results[conf_id]:
            self.results[conf_id][instance_id] = result

    def get_results(self):
        logging.info(f"Publishing results")
        return self.results

    def get_results_single(self, conf_id, instance_id):
        result = False
        if conf_id in list(self.results.keys()):
            if instance_id in list(self.results[conf_id].keys()):
                result = self.results[conf_id][instance_id]
        return result

    def put_start(self,conf_id, instance_id, start):
        logging.info(f"Getting start: {conf_id}, {instance_id}, {start} ")
        if conf_id not in self.start_time:
            self.start_time[conf_id] = {}

        if instance_id not in self.start_time[conf_id]:
            self.start_time[conf_id][instance_id] = start

    def put_race_history(self, tournament):
        self.race_history[tournament.id] = tournament
        self.last_race = tournament

    def put_tournament_update(self, tournament):
        self.races[tournament.id] = tournament

    def remove_tournament(self,tournament):
        self.races.pop(tournament.id)

    def get_termination_single(self, conf_id, instance_id):
        termination = False
        if conf_id in list(self.termination_history.keys()):
            if instance_id in list(self.termination_history[conf_id]):
                termination = True
        return termination

    def save_rt_results(self):
        with open(f"{self.scenario.log_folder}/run_history.json", 'a') as f:
            history = {str(k):v for k,v in self.results.items()}
            json.dump(history, f, indent=2)

    def save_tournament_history(self):
        with open(f"{self.scenario.log_folder}/tournament_history.json", 'a') as f:
            history = {str(k): v for k, v in self.race_history.items()}
            json.dump(history, f, indent=4, cls=TournamentEncoder)

