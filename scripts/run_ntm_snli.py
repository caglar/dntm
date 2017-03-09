import argparse
import numpy as np

import theano
import cPickle as pkl
import warnings

import theano
import theano.tensor as TT
from train_ntm_adam_on_snli import search_model_adam


parser = argparse.ArgumentParser("DNTM on the copy task.")
parser.add_argument("--lr", default=2e-3, type=float)
parser.add_argument("--nhids", default=300, type=int)
parser.add_argument("--std", default=0.05, type=float)
parser.add_argument("--correlation_ws", default=0., type=float)
parser.add_argument("--mem_size", default=10, type=int)
parser.add_argument("--mem_nel", default=128, type=int)
parser.add_argument("--save_path", default=".", type=str)
parser.add_argument("--address_size", default=6, type=int)
parser.add_argument("--renormalization_scale", default=3., type=float)
args = parser.parse_args()

state = {}

state['lr'] = args.lr
state['batch_size'] = 32
state['sub_mb_size'] = None
state['data_path'] = '/data/lisatmp4/gulcehrc/data/snli/SNLI_data.pkl'
state['glove_emb_path'] = '/data/lisatmp4/gulcehrc/data/snli/SNLI_GLoVe_embs.pkl'
state['learn_embeds'] = False
state['use_layer_norm'] = True
state['recurrent_dropout_prob'] = 0.1

state['std'] = args.std
state['max_iters'] = 80000
state['n_hids'] = args.nhids
state['mem_size'] = 64 #args.mem_size
state['mem_nel'] = args.mem_nel
state['renormalization_scale'] = None
state['use_reinforce_baseline'] = True
state['use_reinforce'] = True

state['l1_pen'] = 8e-6
state['l2_pen'] = 6e-5

state['use_ff_controller'] = False
state['bow_size'] = 300
state['n_reading_steps'] = 1
state['n_read_heads'] = 1

state['address_size'] = args.address_size
state['path'] = "/Tmp/gulcehrc/model_files/"
state['save_path'] = state['path']
state['debug'] = False

state['use_loc_based_addressing'] = False
state['use_batch_norm'] = False
state['use_quad_interactions'] = False
state['learn_h0'] = True

np.random.seed(7)
search_model_adam(state, channel=None)
