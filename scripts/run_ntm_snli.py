import argparse
import numpy as np

import theano
import cPickle as pkl
import warnings

import theano
import theano.tensor as TT
from train_ntm_adam_on_snli import search_model_adam


parser = argparse.ArgumentParser("NTM on the SNLI task.")

parser.add_argument("--lr", default=2e-3, type=float)
parser.add_argument("--nhids", default=300, type=int)
parser.add_argument("--std", default=0.05, type=float)
parser.add_argument("--correlation_ws", default=0., type=float)
parser.add_argument("--mem-size", default=10, type=int)
parser.add_argument("--mem-nel", default=128, type=int)
parser.add_argument("--save-path", default=".", type=str)
parser.add_argument("--address-size", default=0, type=int)
parser.add_argument("--renormalization_scale", default=3., type=float)
parser.add_argument("--emb-scale", default=0.32, type=float)
parser.add_argument("--batch-size", default=32, type=int)
args = parser.parse_args()

state = {}

state['lr'] = args.lr

state['batch_size'] = args.batch_size
state['sub_mb_size'] = None
state['data_path'] = '/rap/jvb-000-aa/gulcehre/data/snli/SNLI_data.pkl'
state['glove_emb_path'] = '/rap/jvb-000-aa/gulcehre/data/snli/SNLI_GLoVe_embs.pkl'
state['learn_embeds'] = False
state['use_layer_norm'] = True
state['recurrent_dropout_prob'] = 0.1
state['emb_scale'] = args.emb_scale
state['std'] = args.std

state['max_iters'] = 80000
state['n_hids'] = args.nhids
state['mem_size'] = args.mem_size
state['mem_nel'] = args.mem_nel
state['renormalization_scale'] = None
state['use_reinforce_baseline'] = False
state['use_reinforce'] = False
state['use_loc_based_addressing'] = True
state['use_quad_interactions'] = False

state['l1_pen'] = 0
state['l2_pen'] = 0.

state['use_ff_controller'] = False
state['bow_size'] = 300
state['n_reading_steps'] = 1
state['n_read_heads'] = 1

state['address_size'] = args.address_size
state['path'] = "/rap/jvb-000-aa/gulcehre/dntm_snli/"
state['save_path'] = state['path']
state['debug'] = False

state['use_loc_based_addressing'] = False
state['use_batch_norm'] = False
state['use_quad_interactions'] = False
state['learn_h0'] = True

np.random.seed(7)
search_model_adam(state, channel=None)
