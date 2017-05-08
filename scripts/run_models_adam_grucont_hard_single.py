import theano
import cPickle as pkl
import warnings

import numpy as np

from jobman import DD, flatten, api0, sql

import theano
import theano.tensor as TT
import train_model_adam
import sys

sys.path.append("../codes/")

from core.nan_guard import NanGuardMode

state = DD()

state.lr = 3e-3
state.batch_size = 160
state.sub_mb_size = 160
state.std = 0.05
state.max_iters = 40000
state.n_hids = 240
state.mem_nel = 150
state.mem_size = 28
state.renormalization_scale = 5.0
state.bowout = True
state.use_ff_controller = False
state.std = 0.01
state.bow_size = 80
state.n_reading_steps = 1
state.n_read_heads = 1
state.max_seq_len = 300
state.max_fact_len = 15
state.use_reinforce_baseline = False
state.use_reinforce = True
state.debug = False
state.address_size = 24
state.path = "/scratch/jvb-000-aa/gulcehre/tasks_1-20_v1-2/en-10k/"
state.lambda1_rein = 3e-5
state.lambda2_rein = 1e-5
state.theano_function_mode = None #NanGuardMode(nan_is_error=True, inf_is_error=True)
state.print_every = 50
np.random.seed(3)


#Change the table name everytime you try
TABLE_NAME = "run_grusoft_model_search_3steps_soft_v6"

# You should have an account for jobman
ind = 0

state.task_id = 2
train_model_adam.search_model_adam(state, channel=None)
