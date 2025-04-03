# OPTICAT
This is an implementation of OPTICAT (Optimization of Parameters Through Instance-Specific Configuration and Advantage Training), a configurator based on reinforcement learning for instance-specific algorithm configuration.

The data (instances, scenarios, arguments, wrapper) used for our experiments can be found [here](TPDP)


## Installation
Install the requirements
```
pip install -r requirements.txt
```

To run configurations in parallel, we use [ray](https://www.ray.io). We require Ray version 2.3.1. By default, OPTICAT will make use of all available cores on a machine. To run it on a cluster, we provide a Slurm script that starts Ray before calling OPTICAT.



## Running OPTICAT

### Training
Run OPTICAT by:
```
python RLS/main.py --scenario_file <path_to_your_scenario> --arg_file <path_to_your_argument_file>
```

### Arguments

+ `scenario_file`: Specify the scenario file path
+ `arg_file`: File with other parameters 

Examples of how to run OPTICAT can be found here: [here](XXX)


#### Scenario
The scenario file should contain the following arguments:
+ `cutoff_time`: Time the target algorithm is allowed to run
+ `instance_file`: File containing the paths to training instances
+ `test_instance_file`: File containing the paths to the test instances
+ `feature_file`: File containing the paths to the features of the provided training and test instances
+ `paramfile`: Parameter file of the target algorithm in PCS format
+ `run_obj`[quality, runtime]: Specify the objective for running

The `arg_file` can contain the options below:

 #### General
+ `seed`: Seed to be used by OPTICAT
+ `log_folder`: Path to a folder where OPTICAT should write the logs and results
+ `localmode`: If set to true, Ray will start in local mode, i.e., not look for a running Ray instance in the background
+ `num_cpu`: Number of CPUs to be used. By default this is equal to the race size


#### Run target algorithm
To run the target algorithm, a wrapper is needed that, given the configuration, instance, and seed, returns a command string calling the target algorithm:
+ `wrapper_mod_name`: Path to the wrapper file. The wrapper file will be imported by OPTICAT.
+ `wrapper_class_name`: The class in wrapper_mod_name that should be called by OPTICAT to generate the command for the target algorithm.
+ `memory_limit`: Memory limit for a target algorithm run

# todo add example for quality with quality match

- `par`: PAR penalty score to apply when configuring for runtime.
- `quality_penalty`: The penalty to be applied in case a target algorithm run does not yield any costs when configuring for costs.

- `termination_criterion` [total_runtime, total_race_number]: Stopping criteria.
- `wallclock_limit`: Time OPTICAT is allowed to run when the stopping criterion is `total_runtime`.
- `race_max`: Maximum number of races/iterations OPTICAT is allowed to run when the stopping criterion is `total_race_number`.

- `racesize`: Number of configurations sampled for each instance.
- 
#### Instance set
+ `initial_instance_set_size`: Number of instances sampled in each iteration

#### RL Related

- `rl` [beta, normal]: Distribution to use for learning the continuous parameters.
- `norm` [fnorm, znorm]: Advantage normalization method to be applied.
- `relu_advantage`: If True, the advantage is clipped to only positive values.
- `entropy_loss`: If True, entropy loss is added to the objective with the value of `ec`.
- `ec`: Entropy loss coefficient.
- `num_blocks`: Number of ResNet blocks for the actor.
- `num_neurons`: Number of neurons per layer.
- `clip_value`: Value for gradient clipping.
- `lr`: Learning rate.
- `lr_schedule`: If True, a learning rate schedule is used.
- `init_weights` [uniform, xavier, none]: Initialization method for the actor. 
- `delete_features`: If True, static features are deleted.
- `batchsize`: Batch size.
- `ppo_epochs`: Number of PPO steps to perform in each iteration. 

### Testing
By default, the training script also provides configurations for the training and test sets that can be used. To get additional output, use:
```
python RLS/run_productions.py --scenario_file <path_to_your_scenario> --arg_file <path_to_your_argument_file>
```




Example:
python RLS/main.py --scenario_file "/Users/Elias/Desktop/PhD Code/rl4ac/input/scenarios/rastrigin_example/scenario.txt" --file "/Users/Elias/Desktop/PhD Code/rl4ac/input/scenarios/rastrigin_example/args.txt" --quality_match "output:\s*\d+(\.\d+)?" --quality_extract "(\d+\.\d+)" --feature_free True


python RLS/run_productions.py --scenario_file "/Users/Elias/Desktop/PhD Code/rl4ac/input/scenarios/rastrigin_example/scenario.txt" --file "/Users/Elias/Desktop/PhD Code/rl4ac/input/scenarios/rastrigin_example/args.txt" --feature_free True









