"""This modules includes functions to execute the target algorithm."""


import logging
import time
import subprocess
import ray
import numpy as np
import psutil
import os
import signal
import traceback
import re
import copy

from threading import Thread
from queue import Queue, Empty
from RLS.util.generators.default_point_generator import check_conditionals


def enqueue_output(out, queue):
    """Enqueue output."""
    for line in iter(out.readline, b''):
        line = line.decode("utf-8")
        queue.put(line)
    out.close()


def get_running_processes(ta_process_name):
    """Get list of running processes."""
    processes = []
    for proc in psutil.process_iter():
        try:
            processName = proc.name()
            processID = proc.pid
            if processName in [ta_process_name]:
                processes.append([processName, processID])
        except (psutil.NoSuchProcess, psutil.AccessDenied,
                psutil.ZombieProcess):
            pass
    return processes


def termination_check(process_pid, process_status, ta_process_name, python_pid,
                      conf_id, instance):
    """Check if process was terminated."""
    running_processes = get_running_processes(ta_process_name)

    sr = False
    for rp in running_processes:
        if process_pid == rp[1]:
            sr = True

    if sr:
        logging.info(f"Failed to terminate {conf_id}, {instance}: process {process_pid} with {process_status} on {python_pid} is still running")
    else:
        logging.info(
            f"Successfully terminated {conf_id}, {instance} on {python_pid} with {process_status}")


@ray.remote(num_cpus=1)
def tae_from_cmd_wrapper_rt(conf, instance_path, cache, ta_command_creator, scenario):
    """
    Execute the target algorithm with a given conf/instance pair by calling a user provided Wrapper that created a cmd
    line argument that can be executed
    :param conf: Configuration
    :param instance: Instances
    :param cache: Cache
    :param ta_command_creator: Wrapper that creates a
    :return:
    """
    # todo logging dic should be provided somewhere else -> DOTAC-37
    logging.basicConfig(
        filename=f'./{scenario.log_folder}/wrapper_log_for{conf.id}.log',
        level=logging.INFO, format='%(asctime)s %(message)s')

    try:
        cutoff = scenario.cutoff_time
        logging.info(f"Wrapper TAE start {conf}, {instance_path}")
        runargs = {'instance': f'{instance_path}',
                   'seed': np.random.randint(0,10000),
                   "id": f"{conf.id}"}


        clean_conf = copy.copy(conf.conf)
        # Check conditionals and turn off parameters if violated
        cond_vio = check_conditionals(scenario, clean_conf)
        for cv in cond_vio:
            clean_conf.pop(cv, None)
        cmd = ta_command_creator.get_command_line_args(runargs, clean_conf)
        start = time.time()
        cache.put_start.remote(conf.id, instance_path, start)

        p = psutil.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, close_fds=True)

        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()

        timeout = False
        memory_p = 0
        cpu_time_p = 0
        reading = True

        while reading:
            try:
                line = q.get(timeout=.5)
                empty_line = False
                # Get the cpu time and memory of the process
            except Empty:
                empty_line = True
                if p.poll() is None:
                    cpu_time_p = p.cpu_times().user
                    memory_p = p.memory_info().rss / 1024 ** 2
                if float(cpu_time_p) > float(cutoff) or float(memory_p) > float(scenario.memory_limit) and timeout ==False:
                    timeout = True
                    logging.info(f"Timeout or memory reached, terminating: {conf}, {instance_path} {time.time() - start}")
                    print(f"Timeout or memory reached, terminating: {conf}, {instance_path} {time.time() - start}")
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        print("Got sleep interupt", conf, instance_path)
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    # if scenario.ta_pid_name is not None:
                    #    termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(),conf.id, instance_path)
                pass

            if p.poll() is None:
                # Get the cpu time and memory of the process
                cpu_time_p = p.cpu_times().user
                memory_p = p.memory_info().rss / 1024 ** 2

                if float(cpu_time_p) > float(cutoff) or float(memory_p) > float(
                        scenario.memory_limit) and timeout is False:
                    timeout = True
                    logging.info(f"Timeout or memory reached, terminating: {conf}, {instance_path} {time.time() - start}")
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        print("Got sleep interupt", conf, instance_path)
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    # if scenario.ta_pid_name is not None:
                    #    termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(),conf.id, instance_path)

            # Break the while loop when the ta was killed or finished
            if empty_line and p.poll() is not None:
                reading = False

        if timeout:
            cache.put_result.remote(conf.id, instance_path, np.nan)
        else:
            cache.put_result.remote(conf.id, instance_path, cpu_time_p)

        time.sleep(0.2)
        logging.info(f"Wrapper TAE end {conf}, {instance_path}")
        return conf, instance_path, False

    except KeyboardInterrupt:
        logging.info(f" Killing: {conf}, {instance_path} ")
        # We only terminated the subprocess in case it has started (p is defined)
        if 'p' in vars():
            if p.poll() is None:
                p.terminate()
            try:
                time.sleep(1)
            except:
                print("Got sleep interupt", conf, instance_path)
                pass
            if p.poll() is None:
                p.kill()
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception as e:
                pass
            # if scenario.ta_pid_name is not None:
            #   termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(), conf.id, instance_path)
        cache.put_result.remote(conf.id, instance_path, np.nan)
        try:
            logging.info(f"Killing status: {p.poll()} {conf.id} {instance_path}")
        except:
            pass
        return conf, instance_path, True
    except Exception:
        print({traceback.format_exc()})
        logging.info(f"Exception in TA execution: {traceback.format_exc()}")

@ray.remote(num_cpus=1)
def tae_from_cmd_wrapper_quality(conf, instance_path, cache, ta_command_creator, scenario):
    """
    Execute the target algorithm with a given conf/instance pair by calling a user provided Wrapper that created a cmd
    line argument that can be executed
    :param conf: Configuration
    :param instance: Instances
    :param cache: Cache
    :param ta_command_creator: Wrapper that creates a
    :return:
    """
    logging.basicConfig(filename=f'./{scenario.log_folder}/wrapper_log_for{conf.id}.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')

    try:
        logging.info(f"Wrapper TAE start {conf}, {instance_path}")
        runargs = {'instance': f'{instance_path}',
                   'seed': np.random.randint(0,10000),
                   "id": f"{conf.id}", "timeout": scenario.cutoff_time}

        clean_conf = copy.copy(conf.conf)
        # Check conditionals and turn off parameters if violated
        cond_vio = check_conditionals(scenario, clean_conf)
        for cv in cond_vio:
            clean_conf.pop(cv, None)

        cmd = ta_command_creator.get_command_line_args(runargs, conf.conf)
        start = time.time()
        cache.put_start.remote(conf.id, instance_path, start)

        p = psutil.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, close_fds=True)
        print(cmd)
        # Blocks
        # for line in iter(p.stdout.readline, ''):
        #     line = line.decode("utf-8")
        #    print(line)
        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()

        timeout = False
        empty_line = False
        memory_p = 0
        cpu_time_p = 0
        reading = True
        quality = [np.nan]

        while reading:
            try:
                line = q.get(timeout=.5)
                empty_line = False
                # Get the cpu time and memory of the process
                if scenario.run_obj == "quality":
                    output_tigger = re.search(scenario.quality_match, line)
                    if output_tigger:
                        possible_quality = re.findall(f"{scenario.quality_extract}", line)
                        if len(possible_quality) != 0:
                            quality = possible_quality
            except Empty:
                empty_line = True
                if p.poll() is None:
                    cpu_time_p = p.cpu_times().user
                    memory_p = p.memory_info().rss / 1024 ** 2
                if float(cpu_time_p) > float(scenario.cutoff_time) or float(memory_p) > float(
                        scenario.memory_limit) and timeout == False:
                    timeout = True
                    logging.info(
                        f"Timeout or memory reached, terminating: {conf}, {instance_path} {time.time() - start}")
                    print(f"Timeout or memory reached, terminating: {conf}, {instance_path} {time.time() - start}")
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        print("Got sleep interupt", conf, instance_path)
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    # if scenario.ta_pid_name is not None:
                    #    termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(),conf.id, instance_path)
                pass

            if p.poll() is None:
                # Get the cpu time and memory of the process
                cpu_time_p = p.cpu_times().user
                memory_p = p.memory_info().rss / 1024 ** 2

                if float(cpu_time_p) > float(scenario.cutoff_time) or float(memory_p) > float(
                        scenario.memory_limit) and timeout is False:
                    timeout = True
                    logging.info(f"Timeout or memory reached, terminating: {conf}, {instance_path} {time.time() - start}")
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        print("Got sleep interupt", conf, instance_path)
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    # if scenario.ta_pid_name is not None:
                    #    termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(),conf.id, instance_path)

            # Break the while loop when the ta was killed or finished
            if empty_line and p.poll() is not None:
                reading = False

        cache.put_result.remote(conf.id, instance_path, float(quality[0]))

        time.sleep(0.2)
        logging.info(f"Wrapper TAE end {conf}, {instance_path}")
        return conf, instance_path, False
    except Exception:
        logging.info(f"Exception in TA execution: {traceback.format_exc()}")



@ray.remote(num_cpus=1)
def dummy_task(conf, instance_path, cache):
    time.sleep(2)
    cache.put_result.remote(conf.id, instance_path, np.nan)
    return conf, instance_path, True


