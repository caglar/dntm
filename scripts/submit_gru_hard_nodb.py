import theano
import cPickle as pkl
import warnings

import argparse
import numpy as np

from jobman import DD, flatten, api0, sql

import theano
import theano.tensor as TT
import train_model_adam
from train_model_adam import search_model_adam

state = DD()
parser = argparse.ArgumentParser("Parameters for the single soft model.")
parser.add_argument("--task_id", default=1, type=int)
parser.add_argument("--reload_model", default=1, type=int)
parser.add_argument("--save_path", default=".", type=str)
parser.add_argument("--seed", default=".", type=str)


args = parser.parse_args()
state.reload_model = args.reload_model
state.task_id = args.task_id
state.save_path = args.save_path


state.lr = 3e-3
state.batch_size = 160
state.sub_mb_size = 160
state.max_iters = 40000
state.n_hids = 180
state.mem_nel = 120
state.mem_size = 28
state.renormalization_scale = 5.0
state.use_ff_controller = False
state.std = 0.03
state.bow_size = 100
state.bow_weight_start = 0.64
state.learn_h0 = True

state.seed = args.seed

state.use_gru_inp_rep = True
state.use_bow_input = False
state.bowout = True

state.n_reading_steps = 3
state.n_read_heads = 1
state.max_seq_len = 300
state.max_fact_len = 15
state.use_reinforce_baseline = False
state.use_reinforce = True
state.address_size = 16
state.path = "/rap/jvb-000-aa/data/sarath/en-10k/splitted_trainval/"
#state.path = "/rap/jvb-000-aa/data/sarath/en-10k/"
state.lambda1_rein = 9e-5
state.lambda2_rein = 1e-5

state.l1_reg = 1e-5
state.l2_reg = 5e-5

state.debug = True
state.save_freq = 1000

search_model_adam(state, channel=None, reload_model=state.reload_model)
