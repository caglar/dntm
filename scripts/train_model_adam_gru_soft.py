import math
import os

import theano
import theano.tensor as TT

import numpy as np

import sys
sys.path.append("../codes/")

from core.learning_rule import Adasecant, Adam, RMSPropMomentum, Adasecant2 #, AdaDelta
from core.parameters import (WeightInitializer,
                                    BiasInitializer,
                                    InitMethods,
                                    BiasInitMethods)

from core.nan_guard import NanGuardMode

from core.commons import Tanh, Trect, Sigmoid, Rect, Leaky_Rect
from core.commons import SEEDSetter, DEFAULT_SEED

from memnet.mainloop import FBaBIMainLoop
from memnet.nmodel import NTMModel
from memnet.grumodel import GRUModel
from memnet.fbABIdataiterator import FBbABIDataIteratorSingleQ

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from jobman import DD, flatten, api0, sql

def search_model_adam_gru_soft(state, channel):
    def NReLU(x, rng=None, use_noise=False):
        assert rng is not None
        if use_noise:
            stds = Sigmoid(x)
            x = x + rng.normal(x.shape, avg=0.0, std=stds, dtype=x.dtype)
        return Trect(x)


    def NRect(x, rng=None, use_noise=False, std=0.05):
        assert rng is not None
        if use_noise:
            x = x + rng.normal(x.shape, avg=0.0, std=std, dtype=x.dtype)
        return Trect(x)


    def get_inps(use_mask=True, vgen=None, debug=False):
        if use_mask:
            X, y, mask, cmask = TT.itensor3("X"), TT.imatrix("y"), TT.fmatrix("mask"), TT.fmatrix("cost_mask")
            if debug:
                theano.config.compute_test_value = "warn"
                batch = vgen.next()
                X.tag.test_value = batch['x'].astype("int32")
                y.tag.test_value = batch['y'].astype("int32")
                mask.tag.test_value = batch['mask'].astype("float32")
                cmask.tag.test_value = batch['cmask'].astype("float32")
            return [X, y, mask, cmask]
        else:
            X, y = TT.itensor3("X"), TT.itensor3("y")
            if debug:
                theano.config.compute_test_value = "warn"
                batch = vgen.next()
                X.tag.test_value = batch['x']
                y.tag.test_value = batch['y']
            return [X, y]

    lr = state.lr
    batch_size = state.batch_size
    seed = state.get("seed", 3)

    # No of els in the cols of the content for the memory
    mem_size = state.mem_size

    # No of rows in M
    mem_nel = state.mem_nel
    std = state.std
    renormalization_scale = state.renormalization_scale
    sub_mb_size = state.sub_mb_size

    # No of hids for controller
    n_hids =  state.n_hids

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = state.n_hids

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    seed = 7

    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 1

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 20
    w2v_embed_scale = 0.05

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adam(gradient_clipping=10)
    task_id = state.task_id

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    w2v_embed_path = None

    use_reinforce_baseline = False

    l1_pen = 7e-4
    l2_pen = 9e-4
    debug = False

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/splitted_trainval/"
    prfx = ("ntm_on_fb_BABI_task_all__learn_h0_l1_no_n_hids_%(n_hids)s_bsize_%(batch_size)d"
            "_std_%(std)f_mem_nel_%(mem_nel)d_mem_size_%(mem_size)f_lr_%(lr)f") % locals()

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_train_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_valid_ngram_False.pkl',
                                          max_fact_len=tdata_gen.max_fact_len,
                                          max_seq_len=max_seq_len,
                                          randomize=False,
                                          task_id=task_id,
                                          mode="valid",
                                          task_path=path,
                                          fact_vocab="all_tasks_train_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.Adaptive,
                           center=0.0)

    bi = BiasInitializer(sparsity=-1,
                         scale=std,
                         rng=rng,
                         init_method=BiasInitMethods.Constant,
                         center=0.0)

    print "Length of the vocabulary, ", len(tdata_gen.vocab.items())

    ntm = NTMModel(n_in=len(tdata_gen.vocab.items()),
                   n_hids=n_hids,
                   bow_size=bow_size,
                   n_out=len(tdata_gen.vocab.items()),
                   mem_size=mem_size,
                   mem_nel=mem_nel,
                   use_ff_controller=use_ff_controller,
                   sub_mb_size=sub_mb_size,
                   deep_out_size=deep_out_size,
                   inps=inps,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_gru_inp_rep=False,
                   use_bow_input=True,
                   erase_activ=erase_activ,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=False,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=use_reinforce_baseline,
                   controller_activ=cont_act,
                   use_adv_indexing=False,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   reinforce_decay=0.9,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=100,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=state.max_iters,
                              prefix=prfx)
    main_loop.run()

    return channel.COMPLETE
