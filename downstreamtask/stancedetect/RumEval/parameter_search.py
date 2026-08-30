from hyperopt import fmin, tpe, hp, Trials
import numpy

def parameter_search(ntrials, objective_function, task):
    # فضای جستجو به‌جای generator از لیست استفاده می‌کند
    search_space = {
        'num_dense_layers': [1, 2],  # لیست به‌جای hp.choice
        'num_dense_units': [200, 300, 400, 500],
        'num_epochs': [100, 50],
        'num_lstm_units': [100, 200, 300],
        'num_lstm_layers': [1, 2],
        'learn_rate': [1e-4, 3e-4, 1e-3],
        'mb_size': [32, 64],
        'l2reg': [0.0, 1e-4, 3e-4, 1e-3],
        'rng_seed': [364]
    }

    trials = Trials()
    best = fmin(objective_function,
                space=search_space,
                algo=tpe.suggest,
                max_evals=ntrials,
                trials=trials,
                rstate=numpy.random.RandomState(364))

    print(best)

    # ذخیره بهترین نتیجه
    bp = trials.best_trial['result']['Params']
    f = open('output/trials_' + task + '.txt', "wb")
    pickle.dump(trials, f)
    f.close()

    filename = 'output/bestparams_' + task + '.txt'
    f = open(filename, "wb")
    pickle.dump(bp, f)
    f.close()

    return bp, trials