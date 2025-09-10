# Standard library imports
import sys
import os
import time
import json
import copy
import uuid

import numpy as np
import matplotlib.pyplot as plt
import ray
sys.path.append(os.getcwd())

from RLS.util.ta_result_store import TargetAlgorithmObserver
from RLS.util.ta_execution import dummy_task
from RLS.util.tournament_bookkeeping import (
    get_race_membership,
    update_tasks,
    get_tasks,
    termination_check,
    get_get_tournament_membership_with_ray_id,
)
from RLS.util.log_setup import TournamentEncoder
from RLS.util.tournament_performance import (
    get_instances_no_results,
    get_mean_rt_over_instance,
    get_mean_rt_for_batch,
)
from RLS.util.instance_sets import InstanceSetRACE
from RLS.util.point_gen import PointGen
from RLS.util.generators.default_point_generator import default_point, check_conditionals
from RLS.util.pool import Race

from race_dispatcher import RaceDispatcher
from instance_features import Feature_Transfomer
from ppo_ac_beta import PPO_AC_BETA
from ppo_ac_normal import PPO_AC_NORMAL
from ac_util import smooth_curve, make_prediction, save_to_json, convert_to_serializable



def offline_rl_search_loop(scenario, ta_wrapper, logger):

    if not os.path.exists(f"{scenario.log_folder}/models"):
        os.makedirs(f"{scenario.log_folder}/models")

    if scenario.feature_free is False:
        # Separate training and test instance features
        train_features = {key: scenario.features[key] for key in scenario.instance_set}
        # Prepare data for fitting and transforming
        train_feature_keys = list(train_features.keys())
        train_feature_values = list(train_features.values())
        train_fm = np.array(train_feature_values)

        # Fit the transformer on training features
        scaler = Feature_Transfomer(delete_features=scenario.delete_features)
        train_fm = scaler.fit_transform(train_fm)
        scaler.save_transformer(f"{scenario.log_folder}/models/transformer.pkl")

        # Transform training features
        for fk, fv in zip(train_feature_keys, train_fm):
            scenario.features[fk] = fv
        print(scenario.features)
        if scenario.test_instances:
            # Transform test features using the already fitted transformer
            test_features = {key: scenario.features[key] for key in scenario.test_instances}
            test_feature_keys = list(test_features.keys())
            test_feature_values = list(test_features.values())
            test_fm = np.array(test_feature_values)
            test_fm = scaler.transform(test_fm)
            # Update scenario features with transformed test features
            for fk, fv in zip(test_feature_keys, test_fm):
                scenario.features[fk] = fv

        # get feature dimension
        state_dim = train_fm.shape[1]
    else:
        state_dim = 10

    # setup race and ta managment
    race_dispatcher = RaceDispatcher()
    global_cache = TargetAlgorithmObserver.remote(scenario)

    tasks = []
    races = []
    bug_handel = []
    race_counter = 0

    metrics = {
        "episode_reward": [],
        "episode_loss": [],
        "episode_clip_fraction": [],
        "episode_kl": [],
        "episode_entropy": [],
        "episode_grad_norm": [],
    }

    # init agent
    if scenario.rl =="beta":
        rl_agent = PPO_AC_BETA(scenario.parameter, state_dim, scenario)
    elif scenario.rl =="normal":
        rl_agent = PPO_AC_NORMAL(scenario.parameter, state_dim, scenario)

    rl_agent.init_weights(scenario.init_weights)

    # run the default for the default advantage mode
    if scenario.v_mode =="default":
        default_point_generator = PointGen(scenario, default_point)
        default_conf = default_point_generator.point_generator()

        default_assignment = [[default_conf, i] for i in scenario.instance_set]
        dummy_race = Race(uuid.uuid4(), [], [],[], {}, -1)

        tasks = update_tasks(tasks, default_assignment, dummy_race, global_cache, ta_wrapper, scenario)

        _ = ray.get(tasks)

        default_result = {i:ray.get(global_cache.get_results_single.remote(default_conf.id, i)) for i in scenario.instance_set}

        if scenario.run_obj == "runtime":
            default_result = {k: (v if not np.isnan(v) else -scenario.cutoff_time) for k, v in default_result.items()}
        else:
            default_result = {k: (v if not np.isnan(v) else -scenario.quality_penalty) for k, v in default_result.items()}

    # init the local result store
    results = ray.get(global_cache.get_results.remote())

    # init the instance sampler
    instance_selector = InstanceSetRACE(scenario.instance_set, scenario.instance_set_size, set_size=len(scenario.instance_set), target_reach=0, instance_increment_size=0)

    # creating the first race and adding first conf/instance pairs to ray tasks
    instance_id, instances = instance_selector.get_subset(0)

    # get instance configuration pairs
    features = [scenario.features[i] for i in instances for c in range(scenario.racesize)]
    instance_cpu = [i for i in instances for c in range(scenario.racesize)]
    actions = rl_agent.return_conf(features)
    conf_i_assigments = [[actions[i], instance_cpu[i]] for i in range(len(actions))]

    # set up race to run
    race, initial_assignments = race_dispatcher.init_race(conf_i_assigments,instance_id, scenario.num_cpu)
    races.append(race)
    global_cache.put_tournament_update.remote(race)
    tasks = update_tasks(tasks, initial_assignments, race, global_cache, ta_wrapper, scenario)

    logger.info(f"Initial Race {races}")
    logger.info(f"Initial Tasks, {[get_tasks(o.ray_object_store, tasks) for o in races]}")

    main_loop_start = time.time()

    while termination_check(scenario.termination_criterion, main_loop_start, scenario.wallclock_limit,scenario.race_max, race_counter):

        # wait until we get ta feedback
        winner, not_ready = ray.wait(tasks)
        tasks = not_ready
        try:
            result = ray.get(winner)[0]
            result_conf, result_instance, cancel_flag = result[0], result[1], result[2]

        # Some time a ray worker may crash. We handel that here. I.e if the TA did not run to the end, we reschedule
        except (ray.exceptions.WorkerCrashedError, ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError) as e:
            logger.info(f'Crashed TA worker, {time.ctime()}, {winner}, {e}')
            # Figure out which tournament conf. belongs to
            for r in races:
                conf_instance = get_tasks(r.ray_object_store, winner)
                if len(conf_instance) != 0:
                    race_of_c_i = r
                    break

            conf = [conf for conf in race_of_c_i.c_i_pairs_nr if conf.id == conf_instance[0][0]][0]
            instance = conf_instance[0][1]
            # We check if we have killed the conf and only messed up the termination of the process

            termination_check_c_i = ray.get(global_cache.get_termination_single.remote(conf.id , instance))
            if termination_check_c_i:
                result_conf = conf
                result_instance = instance
                cancel_flag = True
                global_cache.put_result.remote(result_conf.id, result_instance, np.nan)
                logger.info(f"Canceled task with no return: {result_conf}, {result_instance}")
                print(f"Canceled task with no return: {result_conf}, {result_instance}")
            else:  # got no results: need to rescheulde
                next_task = [[conf, instance]]
                tasks = update_tasks(tasks, next_task, race_of_c_i, global_cache, ta_wrapper, scenario)
                logger.info(f"We have no results: rescheduling {conf.id}, {instance} {[get_tasks(o.ray_object_store, tasks) for o in races]}")
                print(f"We have no results: rescheduling {conf.id}, {instance} {[get_tasks(o.ray_object_store, tasks) for o in races]}")
                continue


        # Getting the tournament of the first task id
        if len(tasks) > 0:
            first_task = tasks[0]
            ob_t = get_get_tournament_membership_with_ray_id(first_task, races)

            # Figure out if the tournament of the first task is stale. If so cancel the task and start dummy task.
            if len(ob_t.c_i_pairs_nr) == 1:
                i_no_result = get_instances_no_results(results, ob_t.c_i_pairs_nr[0][0].id, ob_t.instance_set)
                if len(i_no_result) == 1:
                    termination = ray.get(global_cache.get_termination_single.remote(ob_t.c_i_pairs_nr[0][0].id, i_no_result[0]))
                    result = ray.get(global_cache.get_results_single.remote(ob_t.c_i_pairs_nr[0][0].id, i_no_result[0]))
                    if termination and result == False and [ob_t.c_i_pairs_nr[0],i_no_result[0]] not in bug_handel:
                        logger.info(f"Stale tournament: {time.strftime('%X %x %Z')}, {ob_t.c_i_pairs_nr[0]}, {i_no_result[0]} , {first_task}, {bug_handel}")
                        print(f"Stale tournament: {time.strftime('%X %x %Z')}, {ob_t.c_i_pairs_nr[0]}, {i_no_result[0]} , {first_task}, {bug_handel}")
                        ready_ids, _remaining_ids = ray.wait([first_task], timeout=0)
                        if len(_remaining_ids) == 1:
                            ray.cancel(first_task)
                            tasks.remove(first_task)
                            task = dummy_task.remote(ob_t.c_i_pairs_nr[0],i_no_result[0], global_cache)
                            tasks.append(task)
                            bug_handel.append([ob_t.c_i_pairs_nr[0],i_no_result[0]])


        # get the results from the store
        if result_conf.id in list(results.keys()):
            results[result_conf.id][result_instance] = ray.get(global_cache.get_results_single.remote(result_conf.id,result_instance ))
        else:
            results[result_conf.id]= {}
            results[result_conf.id][result_instance] = ray.get(global_cache.get_results_single.remote(result_conf.id,result_instance ))

        result_race = get_race_membership(races, result_conf)

        # Check whether we canceled a task or if the TA terminated regularly
        # In case we canceled a task, we need to remove it from the ray tasks
        # This is needed if we want to add capping later
        if cancel_flag:
            if result_conf.id in result_race.ray_object_store.keys():
                if result_instance in result_race.ray_object_store[result_conf.id ].keys():
                    if result_race.ray_object_store[result_conf.id][result_instance] in tasks:
                        tasks.remove(result_race.ray_object_store[result_conf.id][result_instance])
            logger.info(f"Canceled TA: {result_conf.id}, {result_instance}")
        else:
            result_time = results[result_conf.id][result_instance]
            logger.info(f"TA result: {result_conf.id}, {result_instance} {result_time}")

        # Update the race based on result
        result_race, race_stop = race_dispatcher.update_race([result_conf,result_instance], result_race)

        if race_stop:
            print("Iteration:", time.time() - main_loop_start, race_counter)

            race_counter += 1
            start_update = time.time()
            # Get the instances for the new race
            instance_id, instances = instance_selector.get_subset(result_race.instance_set_id + 1)

            # Iterate over all UUID keys in the results dictionary and check if NAs were produced
            print(f"Results.items(): {results.items()}")
            #for uuid_key, inner_dict in results.items():
            #    for key, value in inner_dict.items():
            #        if value is None or (isinstance(value, float) and np.isnan(value)):
            #            raise ValueError(f"The target algorithm command was incorrectly set: NA values were produced.")


            # wenn scenario quality gesetzt ist und alle NANs sind, Fehler ausgeben: target algorithmus-command nicht korrekt
            # Für runtime: runtime für alle < 1 Sekunde? Dann warning ausgeben (wrapper evtl nicht korrekt)
            # Problem:  python "C:/Users/Admin/Desktop/opticat/input/ta/griewank/griewank.py"  --x1=3.6223513793945314 --x2=1.7205224609375 --seed=7126
            # gibt NANs in der Liste zurück
            races.remove(result_race)
            global_cache.put_race_history.remote(result_race)

            # prep rewards for ppo
            if scenario.run_obj == "runtime":
                rewards = [-scenario.cutoff_time * scenario.par if np.isnan(results[c_i[0]][c_i[1]]) else -results[c_i[0]][c_i[1]] for c_i in result_race.c_i_order]
            else:
                rewards = [-scenario.quality_penalty if np.isnan(results[c_i[0]][c_i[1]]) else -results[c_i[0]][c_i[1]] for c_i in result_race.c_i_order]

            # prep v for ppo
            if scenario.v_mode =="default":
                v = [-default_result[c_i[1]] for c_i in result_race.c_i_order]
            elif scenario.v_mode == "mean":
                if scenario.run_obj == "runtime":
                    v = [-get_mean_rt_over_instance(results,[c_i[1]],scenario.cutoff_time, scenario.par )[c_i[1]] for c_i in result_race.c_i_order]
                else:
                    v = [-get_mean_rt_over_instance(results,[c_i[1]],scenario.quality_penalty, 1 )[c_i[1]] for c_i in result_race.c_i_order]
            elif scenario.v_mode == "mean_batch":
                if scenario.run_obj == "runtime":
                    unique_confs = set(pair[0] for pair in result_race.c_i_order)
                    unique_instances = set(pair[1] for pair in result_race.c_i_order)
                    batch_mean = get_mean_rt_for_batch(results, unique_confs, unique_instances, scenario.cutoff_time, scenario.par)
                    v = [-batch_mean[c_i[1]] for c_i in result_race.c_i_order]
                else:
                    unique_confs = set(pair[0] for pair in result_race.c_i_order)
                    unique_instances = set(pair[1] for pair in result_race.c_i_order)
                    batch_mean = get_mean_rt_for_batch(results, unique_confs, unique_instances, scenario.quality_penalty,1)
                    v = [-batch_mean[c_i[1]] for c_i in result_race.c_i_order]
            else:
                raise ValueError("V Mode not defined")

            # update agent
            loss, clip_fraction, kl, grad_norm, entropy = rl_agent.train(rewards, v)

            current_metrics = {
                "episode_reward": sum(rewards) / len(rewards),
                "episode_loss": loss,
                "episode_clip_fraction": clip_fraction,
                "episode_kl": kl,
                "episode_entropy": entropy,
                "episode_grad_norm": grad_norm,
            }

            for key, value in current_metrics.items():
                metrics[key].append(value)

            # conf instance assigment for next race
            features = [scenario.features[i] for i in instances for c in range(scenario.racesize)]
            instance_cpu = [i for i in instances for c in range(scenario.racesize)]
            actions = rl_agent.return_conf(features)
            conf_i_assigments = [[actions[i], instance_cpu[i]] for i in range(len(actions))]

            # Create new race
            new_race, initial_assignments_new_tournament = race_dispatcher.init_race(conf_i_assigments,instance_id, scenario.num_cpu)

            # Add the new tournament and update the ray tasks with the new conf/instance assignments
            races.append(new_race)
            tasks = update_tasks(tasks, initial_assignments_new_tournament, new_race, global_cache,  ta_wrapper, scenario)

            global_cache.put_tournament_update.remote(new_race)
            global_cache.remove_tournament.remote(result_race)
            print("Update time", time.time()-start_update)
            logger.info(f"Final results tournament {result_race}")
            logger.info(f"New tournament {new_race}")

            rl_agent.save_model(f"{scenario.log_folder}/models/actor_{race_counter}.pth")
        else:
            # If the race does not terminate we get a new conf/instance assignment and add that as ray task
            next_task = race_dispatcher.next_race_run(result_race)
            tasks = update_tasks(tasks, next_task, result_race, global_cache, ta_wrapper, scenario)
            global_cache.put_tournament_update.remote(result_race)

    # get the final confs for the features
    # Process training instances
    make_prediction(
        instances=scenario.instance_set,
        scenario=scenario,
        rl_agent=rl_agent,
        output_file=f"{scenario.log_folder}/final_training.json"
    )

    if scenario.test_instances:
        # Process test instances
        make_prediction(
            instances=scenario.test_instances,
            scenario=scenario,
            rl_agent=rl_agent,
            output_file=f"{scenario.log_folder}/final_test.json"
        )

    rl_agent.save_model(f"{scenario.log_folder}/models/actor_final.pth")

    plt.plot(metrics['episode_loss'])
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{scenario.log_folder}/loss.png")
    plt.close()

    plt.plot(metrics['episode_entropy'])
    plt.title("Critic Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{scenario.log_folder}/entropy.png")
    plt.close()

    # Smooth the curve
    window_size = 5  # Adjust window size as needed
    smoothed_rewards = smooth_curve(metrics['episode_reward'], window_size)
    plt.plot(metrics['episode_reward'])
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label='Smoothed Rewards', linewidth=2)
    plt.title("RewardOver Time")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.savefig(f"{scenario.log_folder}/reward.png")
    plt.close()

    logger.info(f"Loss {metrics['episode_loss']}")
    logger.info(f"Reward {metrics['episode_reward']}")
    logger.info(f"Clip fraction {metrics['episode_clip_fraction']}")
    logger.info(f"KL {metrics['episode_kl']}")
    logger.info(f"Grad norm {metrics['episode_grad_norm']}")
    logger.info(f"Entropy {metrics['episode_entropy']}")


    for metric_name, metric_data in metrics.items():

        # Convert the data before saving it as a JSON file
        cleaned_data = convert_to_serializable(metric_data)

        save_to_json(cleaned_data, f"{scenario.log_folder}/{metric_name}.json")

    global_cache.save_rt_results.remote()
    global_cache.save_tournament_history.remote()

    while not os.path.isfile(f"{scenario.log_folder}/tournament_history.json"):
        time.sleep(60)

    print("DONE")
    logger.info("DONE")
    time.sleep(30)
    [ray.cancel(t) for t in not_ready]
