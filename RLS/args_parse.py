import argparse

class LoadOptionsFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def parse_args():
    """
    Argparser
    :return: dic. Dic of arguments parsed
    """

    parser = argparse.ArgumentParser()
    hp = parser.add_argument_group("Hyperparameters of util")
    so = parser.add_argument_group("Scenario options")

    hp.add_argument('--file', type=open, action=LoadOptionsFromFile)

    hp.add_argument('--seed', default=42, type=int)
    hp.add_argument('--log_folder', type=str, default="latest")
    hp.add_argument('--memory_limit', type=int, default=1023*2)

    hp.add_argument('--ta_run_type', type=str, default="import_wrapper")
    hp.add_argument('--wrapper_mod_name', type=str, default="")
    hp.add_argument('--wrapper_class_name', type=str, default="")

    parser.add_argument('--localmode', default=True, type=lambda x: (str(x).lower() == 'true'))
    hp.add_argument('--termination_criterion', type=str, default="runtime")
    hp.add_argument('--race_max', type=int, default=4)
    hp.add_argument('--num_cpu', type=int)

    hp.add_argument('--racesize', type=int, default=4)
    hp.add_argument('--instance_set_size', type=int, default=5)
    parser.add_argument('--feature_free', default=False, type=lambda x: (str(x).lower() == 'true'))



    hp.add_argument('--par', type=int, default=1)
    hp.add_argument('--quality_penalty', type=int, default=500)
    hp.add_argument('--quality_match', type=str, default="")
    hp.add_argument('--quality_extract', type=str, default="")


    hp.add_argument('--rl', type=str, default="beta")
    hp.add_argument('--norm', type=str, default="fnorm")
    hp.add_argument('--v_mode', type=str, default="default")
    parser.add_argument('--relu_advantage', default=False, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--entropy_loss', default=False, type=lambda x: (str(x).lower() == 'true'))
    hp.add_argument('--ec', type=float, default=0.001)

    hp.add_argument('--num_blocks', type=int, default=9)
    hp.add_argument('--num_neurons', type=int, default=512)
    hp.add_argument('--clip_value', type=float, default=0.5)
    parser.add_argument('--lr_schedule', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--init_weights',default = "xavier", type=str)
    parser.add_argument('--delete_features', default=True, type=lambda x: (str(x).lower() == 'true'))

    hp.add_argument('--batchsize', type=int, default=127)
    hp.add_argument('--lr', type=float, default=1e-4)
    hp.add_argument('--ppo_epochs', type=int, default=10)

    so.add_argument('--scenario_file', type=str)
    so.add_argument('--run_obj', type=str)
    so.add_argument('--cutoff_time', type=str)
    so.add_argument('--wallclock_limit', type=str)
    so.add_argument('--instance_file', type=str)
    so.add_argument('--feature_file', type=str)
    so.add_argument('--paramfile', type=str)
    so.add_argument('--test_instance_file', type=str)


    return vars(parser.parse_args())