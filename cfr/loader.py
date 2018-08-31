import os
import numpy as np

from logger import Logger as Log

def load_result_file(file):
    arr = np.load(file)

    D = dict([(k, arr[k]) for k in arr.keys()])

    return D

def load_config(cfgfile):
    """ Parses a configuration file """

    cfgf = open(cfgfile,'r')
    cfg = {}
    for l in cfgf:
        ps = [p.strip() for p in l.split(':')]
        if len(ps)==2:
            try:
                cfg[ps[0]] = float(ps[1])
            except ValueError:
                cfg[ps[0]] = ps[1]
                if cfg[ps[0]] == 'False':
                    cfg[ps[0]] = False
                elif cfg[ps[0]] == 'True':
                    cfg[ps[0]] = True
    cfgf.close()
    return cfg

def load_single_result(result_dir):
    if Log.VERBOSE:
        print 'Loading %s...' % result_dir

    config_path = '%s/config.txt' % result_dir
    has_config = os.path.isfile(config_path)
    if not has_config:
        print 'WARNING: Could not find config.txt for %s. Skipping.' % os.path.basename(result_dir)
        config = None
    else:
        config = load_config(config_path)

    train_path = '%s/result.npz' % result_dir
    test_path = '%s/result.test.npz' % result_dir

    has_test = os.path.isfile(test_path)

    try:
        train_results = load_result_file(train_path)
    except:
        'WARNING: Couldnt load result file. Skipping'
        return None

    n_rep = np.max([config['repetitions'], config['experiments']])

    if len(train_results['pred'].shape) < 4 or train_results['pred'].shape[2] < n_rep:
        print 'WARNING: Experiment %s appears not to have finished. Skipping.' % result_dir
        return None

    if has_test:
        test_results = load_result_file(test_path)
    else:
        test_results = None

    return {'train': train_results, 'test': test_results, 'config': config}

def load_results(output_dir):

    if Log.VERBOSE:
        print 'Loading results from %s...' % output_dir

    ''' Detect results structure '''
    # Single result
    if os.path.isfile('%s/results.npz' % output_dir):
        #@TODO: Implement
        pass

    # Multiple results
    files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
    exp_dirs = [f for f in files if os.path.isdir(f)
                    if os.path.isfile('%s/result.npz' % f)]

    if Log.VERBOSE:
        print 'Found %d experiment configurations.' % len(exp_dirs)

    # Load each result folder
    results = []
    for dir in sorted(exp_dirs):
        dir_result = load_single_result(dir)
        if dir_result is not None:
            results.append(dir_result)

    return results

def load_train_test_data(data_path_train, data_path_test):
    if data_path_train[-4:] == '.csv':
        if data_path_test is not None: # TODO: currently assumes data_path_test is either None or same as data_path_train, and if not None, test data is last 10% of data_path
            return load_data_csvs(data_path_train, 0.1)
        else:
            return load_data_csvs(data_path_train, 0.0)
    elif data_path_train[-4:] == '.npz':
        if data_path_test is not None:
            return load_data_npz(data_path_train), load_data_npz(data_path_test)
        else:
            return load_data_npz(data_path_train), None

def load_data_npz(datapath):
    """ Load dataset """
    arr = np.load(datapath)
    xs = arr['x']

    HAVE_TRUTH = False
    SPARSE = False

    if len(xs.shape)==1:
        SPARSE = True

    ts = arr['t']
    yfs = arr['yf']
    try:
        es = arr['e']
    except:
        es = None
    try:
        ate = np.mean(arr['ate'])
    except:
        ate = None
    try:
        ymul = arr['ymul'][0,0]
        yadd = arr['yadd'][0,0]
    except:
        ymul = 1
        yadd = 0
    try:
        ycfs = arr['ycf']
        mu0s = arr['mu0']
        mu1s = arr['mu1']
        HAVE_TRUTH = True
    except:
        print 'Couldn\'t find ground truth. Proceeding...'
        ycfs = None; mu0s = None; mu1s = None

    data = {'x':xs, 't':ts, 'e':es, 'yf':yfs, 'ycf':ycfs, \
            'mu0':mu0s, 'mu1':mu1s, 'ate':ate, 'YMUL': ymul, \
            'YADD': yadd, 'ATE': ate.tolist(), 'HAVE_TRUTH': HAVE_TRUTH, \
            'SPARSE': SPARSE}

    return data

def load_data_csvs(data_path_unformatted, test_split):
    """ Load dataset """

    n_reps = 0
    while os.path.isfile(data_path_unformatted % (n_reps + 1)):
        n_reps += 1
    if n_reps == 0:
        raise Exception('No data found at %s' % data_path_unformatted)

    data = {}
    if test_split != 0.0:
        data_test = {}
    else:
        data_test = None

    for i in range(n_reps):
        data_in = np.loadtxt(data_path_unformatted % (i + 1), delimiter=',')
        if i == 0:  # set up arrays
            total_sample_size = data_in.shape[0]
            train_sample_size = int(round(total_sample_size * (1 - test_split)))
            test_sample_size = total_sample_size - train_sample_size
            data['HAVE_TRUTH'] = True  # TODO: detect whether ycf is present
            data['SPARSE'] = False     # TODO: detect whether x is one-dimensional
            data['e'] = None
            data['ate'] = 4
            data['ATE'] = 4
            data['YMUL'] = 1
            data['YADD'] = 0

            data['t'] = np.zeros((train_sample_size, n_reps))
            data['yf'] = np.zeros((train_sample_size, n_reps))
            data['ycf'] = np.zeros((train_sample_size, n_reps))
            data['mu0'] = np.zeros((train_sample_size, n_reps))
            data['mu1'] = np.zeros((train_sample_size, n_reps))
            data['x'] = np.zeros((train_sample_size, data['dim'], n_reps))
            if test_split != 0.0:
                data_test['HAVE_TRUTH'] = True  # TODO: detect whether ycf is present
                data_test['SPARSE'] = False  # TODO: detect whether x is one-dimensional
                data_test['e'] = None
                data_test['ate'] = 4
                data_test['ATE'] = 4
                data_test['YMUL'] = 1
                data_test['YADD'] = 0

                data_test['t'] = np.zeros((test_sample_size, n_reps))
                data_test['yf'] = np.zeros((test_sample_size, n_reps))
                data_test['ycf'] = np.zeros((test_sample_size, n_reps))
                data_test['mu0'] = np.zeros((test_sample_size, n_reps))
                data_test['mu1'] = np.zeros((test_sample_size, n_reps))
                data_test['x'] = np.zeros((test_sample_size, data_test['dim'], n_reps))

        data['t'][:, i] = data_in[:train_sample_size, 0]
        data['yf'][:, i] = data_in[:train_sample_size, 1]
        data['ycf'][:, i] = data_in[:train_sample_size, 2]
        data['mu0'][:, i] = data_in[:train_sample_size, 3]
        data['mu1'][:, i] = data_in[:train_sample_size, 4]
        data['x'][:, :, i] = data_in[:train_sample_size, 5:]
        if test_split != 0.0:
            data_test['t'][:, i] = data_in[train_sample_size:, 0]
            data_test['yf'][:, i] = data_in[train_sample_size:, 1]
            data_test['ycf'][:, i] = data_in[train_sample_size:, 2]
            data_test['mu0'][:, i] = data_in[train_sample_size:, 3]
            data_test['mu1'][:, i] = data_in[train_sample_size:, 4]
            data_test['x'] = data_in[train_sample_size:, 5:]

    return data, data_test
