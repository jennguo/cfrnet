#loads the configs and calls cfr.evaluation.evaluate(...), then saves the results
#from readme: The script evaluate.py performs an evaluation of a trained model based on the predictions made for the training and test sets.

import sys
import os

import cPickle as pickle

from cfr.logger import Logger as Log
Log.VERBOSE = True

import cfr.evaluation as evaluation
from cfr.plotting import *

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted


def load_config(config_file):  														# returns the dictionary defined in the config file
    with open(config_file, 'r') as f:
        # take every line l in the file, then split it by '=' if it has one, and put this tuple/list of 2 into the cfg list so now cfg is a list of pairs
        cfg = [l.split('=') for l in f.read().split('\n') if '=' in l]
        # convert cfg to a dictionary, evaluating the strings (what was on the right of the '=')
        cfg = dict([(kv[0], eval(kv[1])) for kv in cfg])
    return cfg


def evaluate(config_file, overwrite=False, filters=None):
    if not os.path.isfile(config_file): 											# load the configs
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)

    output_dir = cfg['outdir']														# set the output directory, output_dir, according to the config file

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['datadir']+'/'+cfg['dataform']									# set the training and test directories, data_train and data_test, accordingly
    data_test = cfg['datadir']+'/'+cfg['data_test']
    binary = False																	# binary = is the data binary? if log loss function was used then yes
    if cfg['loss'] == 'log':
        binary = True

    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir									# the % is for string formatting, i.e. eval_path = [output_dir]/evaluation.npz
    if overwrite or (not os.path.isfile(eval_path)):								# proceed if you don't need to overwrite anything, or even if you do but you've been given 
																						# permission to overwrite 
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=binary)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))					# save the results
    else:
        if Log.VERBOSE:
            print 'Loading evaluation results from %s...' % eval_path
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))

    # Sort by alpha
    #eval_results, configs = sort_by_config(eval_results, configs, 'p_alpha')

    # Print evaluation results
    if binary:
        plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters)
    else:
        plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)

    # Plot evaluation
    #if configs[0]['loss'] == 'log':
    #    plot_cfr_evaluation_bin(eval_results, configs, output_dir)
    #else:
    #    plot_cfr_evaluation_cont(eval_results, configs, output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:	# the number of parameters provided when this file (evaluate.py) is run from the shell. if it's too few, instruct the user of the appropriate usage.
        print 'Usage: python evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>'
    else:
        config_file = sys.argv[1]

        overwrite = False
        if len(sys.argv)>2 and sys.argv[2] == '1':
            overwrite = True

        filters = None
        if len(sys.argv)>3:
            filters = eval(sys.argv[3])

        evaluate(config_file, overwrite, filters=filters)
