import theano
import cPickle as pkl
import warnings

import numpy as np

#from jobman import DD, flatten, api0, sql

import theano
import theano.tensor as TT
import sys

sys.path.append("../codes/")

import train_model_adam


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

state = Map()

state['lr'] = 1e-4
state['batch_size'] = 32
state['sub_mb_size'] = state['batch_size']
state['std'] = 0.05
state['max_iters'] = 40000
state['n_hids'] = 180
state['debug'] = False
state['mem_nel'] = 120
state['mem_size'] = 28
state['renormalization_scale'] = 3.0
state['use_ff_controller'] = False
state['std'] = 0.05
state['bow_size'] = 100
state['n_reading_steps'] = 3
state['n_read_heads'] = 1
state['max_seq_len'] = 300
state['max_fact_len'] = 15
state['use_layer_norm'] = False
state['use_reinforce_baseline'] = False
state['use_reinforce'] = False
state['address_size'] = 20
state['path'] = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/splitted_trainval/"
state['save_path'] = "/Tmp/gulcehrc/models/"
state['seed'] = 12
n_tasks = 20

np.random.seed(3)


#Change the table name everytime you try
TABLE_NAME = "run_grusoft_model_search_3steps_soft_v3"

# You should have an account for jobman
ind = 0

state['task_id'] = 2
train_model_adam.search_model_adam(state, channel=None)
