import theano
import cPickle as pkl
import warnings

import numpy as np

from jobman import DD, flatten, api0, sql

import theano
import theano.tensor as TT
import memnet.train_model_adam_gru_soft

n_trials = 64
lr_min = 8e-5
lr_max = 1e-2
batches = [100, 200, 400, 800]
renormalization_scale = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
mem_nels = [200, 220, 230, 240, 250, 260, 290, 300]
mem_sizes = [20, 24, 28, 30, 32]
std_min = 0.01
std_max = 0.05

state = DD()

state.lr = 6e-6
state.batch_size = 200
state.sub_mb_size = 25
state.std = 0.01
state.max_iters = 20000
state.n_hids = 200
state.mem_nel = 200
state.mem_size = 28

np.random.seed(3)

ri = np.random.random_integers
learning_rates = np.logspace(np.log10(lr_min), np.log10(lr_max), 100)
stds = np.random.uniform(std_min, std_max, 100)

#Change the table name everytime you try
TABLE_NAME = "adam_grusoft_model_search_v0"

# You should have an account for jobman
db = api0.open_db('postgresql://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=' + TABLE_NAME)
ind = 0

for i in xrange(n_trials):
    state.lr = learning_rates[ri(learning_rates.shape[0]) - 1]
    state.std = stds[ri(len(stds)) - 1]
    state.batch_size = batches[ri(len(batches)) - 1]
    state.renormalization_scale = renormalization_scale[ri(len(renormalization_scale)) - 1]
    state.mem_nel = mem_nels[ri(len(mem_nels)) - 1]
    state.mem_size = mem_sizes[ri(len(mem_sizes)) - 1]
    state.std = stds[ri(stds.shape[0]) - 1]
    sql.insert_job(memnet.train_model_adam_gru_soft.search_model_adam_gru_soft, flatten(state), db)
    ind += 1

db.createView(TABLE_NAME + "_view")
print "{} jobs submitted".format(ind)

