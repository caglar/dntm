import theano
import cPickle as pkl
import warnings

import numpy as np

from jobman import DD, flatten, api0, sql

import theano
import theano.tensor as TT
import sys

sys.path.append("../codes/")

import memnet.train_model_adam

state = DD()

state.lr = 1e-4
state.batch_size = 160
state.sub_mb_size = 80
state.std = 0.05
state.max_iters = 40000
state.n_hids = 180
state.mem_nel = 120
state.mem_size = 28
state.renormalization_scale = 3.0
state.use_ff_controller = False
state.std = 0.05
state.bow_size = 100
state.n_reading_steps = 3
state.n_read_heads = 1
state.max_seq_len = 300
state.max_fact_len = 15
state.use_reinforce_baseline = False
state.use_reinforce = False
state.address_size = 20
state.path = "/scratch/jvb-000-aa/gulcehre/tasks_1-20_v1-2/en-10k/"
state.save_path = "/scratch/jvb-000-aa/gulcehre/models/"
n_tasks = 20
np.random.seed(3)


#Change the table name everytime you try
TABLE_NAME = "run_grusoft_model_search_3steps_soft_v3"

# You should have an account for jobman
ind = 0

state.task_id = 1
memnet.train_model_adam.search_model_adam(state, channel=None)
