import math
import os

import theano
import theano.tensor as TT

import numpy as np
from core.learning_rule import Adasecant, Adam, RMSPropMomentum, Adasecant2 #, AdaDelta
from core.parameters import (WeightInitializer,
                                    BiasInitializer,
                                    InitMethods,
                                    BiasInitMethods)

from core.nan_guard import NanGuardMode

from core.commons import Tanh, Trect, Sigmoid, Rect, Leaky_Rect
from mainloop import FBaBIMainLoop
from nmodel import NTMModel
from fbABIdataiterator import FBbABIDataIteratorSingleQ

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lstmmodel import LSTMModel



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


def get_inps(use_mask=True, vgen=None, use_bow_out=False, debug=False, output_map=False):
    if use_mask:
        X, y, mask, cmask = TT.itensor3("X"), TT.imatrix("y"), TT.fmatrix("mask"), \
                TT.fmatrix("cost_mask")
        qmask = TT.fmatrix("qmask")
        bow_out = TT.ftensor3("bow_out")

        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.next()
            X.tag.test_value = batch['x'].astype("int32")
            y.tag.test_value = batch['y'].astype("int32")
            mask.tag.test_value = batch['mask'].astype("float32")
            cmask.tag.test_value = batch['cmask'].astype("float32")
            qmask.tag.test_value = batch["qmask"].astype("float32")
            if use_bow_out:
                bow_out.tag.test_value = batch['bow_out'].astype("float32")

        if output_map:
            outs = {}
            outs["X"] = X
            outs["y"] = y
            outs["mask"] = mask
            outs["cmask"] = cmask
            if use_bow_out:
                outs["bow_out"] = bow_out
            outs["qmask"] = qmask
        else:
            outs = [X, y, mask, cmask]
            if use_bow_out:
                outs += [bow_out]
            outs += [qmask]
        return outs
    else:
        X, y = TT.itensor3("X"), TT.itensor3("y")
        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.next()
            X.tag.test_value = batch['x']
            y.tag.test_value = batch['y']
        return [X, y]


def get_inps_flat(use_mask=True, vgen=None, debug=False):
    if use_mask:
        X, y, mask, cmask = TT.itensor3("X"), TT.ivector("y"), TT.fmatrix("mask"), TT.fmatrix("cost_mask")
        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.next()
            X.tag.test_value = batch['x'].astype("int32")
            y.tag.test_value = batch['y'].astype("int32")
            mask.tag.test_value = batch['mask'].astype("float32")
            cmask.tag.test_value = batch['cmask'].astype("float32")
        return X, y, mask, cmask
    else:
        X, y = TT.itensor3("X"), TT.ivector("y")
        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.next()
            X.tag.test_value = batch['x']
            y.tag.test_value = batch['y']
        return X, y


def get_inps_e2emem(vgen=None, debug=False):
    X, q, y, mask, cmask = TT.itensor3("X"), TT.imatrix('q'), TT.ivector("y"), \
        TT.fmatrix("mask"), TT.fmatrix("cost_mask")
    if debug:
        assert vgen is not None
        theano.config.compute_test_value = "warn"
        batch = vgen.next()
        X.tag.test_value = batch['x'].astype("int32")
        q.tag.test_value = batch['q'].astype("int32")
        y.tag.test_value = batch['y'].astype("int32")
        mask.tag.test_value = batch['mask'].astype("float32")
        cmask.tag.test_value = batch['cmask'].astype("float32")
    return X, q, y, mask, cmask


def get_inps2(use_mask=True, vgen=None, debug=False):
    if use_mask:
        X, y, mask, qmask, cmask = TT.itensor3("X"), TT.imatrix("y"), TT.fmatrix("mask"), \
                TT.fmatrix("qmask"), TT.fmatrix("cost_mask")
        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.next()
            X.tag.test_value = batch['x'].astype("int32")
            y.tag.test_value = batch['y'].astype("int32")
            #import ipdb; ipdb.set_trace()
            mask.tag.test_value = batch['mask'].astype("float32")
            qmask.tag.test_value = batch['qmask'].astype("float32")
            cmask.tag.test_value = batch['cmask'].astype("float32")
        return X, y, mask, qmask, cmask
    else:
        X, y = TT.itensor3("X"), TT.itensor3("y")
        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.next()
            X.tag.test_value = batch['x']
            y.tag.test_value = batch['y']
        return X, y


def get_powerup_props(rng):
    bi = BiasInitializer(sparsity=-1,
                         scale=0.01,
                         rng=rng,
                         init_method=BiasInitMethods.Constant,
                         center=0.0)

    pi = BiasInitializer(sparsity=-1,
                         scale=0.01,
                         rng=rng,
                         init_method=BiasInitMethods.Constant,
                         center=np.log(np.exp(1) - 1))
    n_pools = 3
    return {"n_pools": n_pools, "bias_init": bi, "pow_init": pi}


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_rec2():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 140

    # No of hids for controller
    n_hids =  100

    #No of cols in M
    mem_size = 20

    #No of rows in M
    mem_nel = 128

    #Not using deep out
    use_deepout = False

    deep_out_size = None

    # Size of the bow embeddings
    bow_size = 50

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True

    # Use loc based addressing:
    use_loc_based_addressing = True

    std = 0.05
    seed = 7

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = True

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7

    l1_pen = 6e-4
    l2_pen = 1e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_3rheads_1whead_tiny_l2pen_simple_small_10k_noadv_rec2"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = False,
                                          max_seq_len = 20,
                                          max_fact_len = 7,
                                          task_id = 1,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = 20,
                                          randomize = False,
                                          task_id = 1,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   controller_activ=cont_act,
                   use_adv_indexing=False,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   seq_len=20,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_rec():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 144

    # No of hids for controller
    n_hids =  90

    # No of cols in M
    mem_size = 32

    # No of rows in M
    mem_nel = 40

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True

    # Use loc based addressing:
    use_loc_based_addressing = True

    std = 0.05
    seed = 7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 10

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7

    l1_pen = 6e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_3rheads_1whead_tiny_l2pen_simple_small_10k_noadv"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = 20,
                                          max_fact_len = 7,
                                          task_id = 1,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = 20,
                                          randomize = False,
                                          task_id = 1,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   w2v_embed_path=w2v_embed_path,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   controller_activ=cont_act,
                   use_adv_indexing=False,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   seq_len=20,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=state.get('print_every', 40),
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_ffcontroller_curriculum_simple_small_10k():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 140

    # No of hids for controller
    n_hids =  100

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 128

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = False

    # Use loc based addressing:
    use_loc_based_addressing = True

    std = 0.05
    seed = 7

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_ffcont"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = 20,
                                          max_fact_len = 7,
                                          task_id = 1,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = 20,
                                          randomize = False,
                                          task_id = 1,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   controller_activ=cont_act,
                   use_adv_indexing=False,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   seq_len=20,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_ffcontroller_curriculum_simple_small_10k_hard_all():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 100

    # No of hids for controller
    n_hids =  180

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 200

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 20

    task_id = None

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    max_fact_len = 16
    max_seq_len = 240

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_ffcont_hard_rav_t20_all"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   use_reinforce=True,
                   use_reinforce_baseline=True,
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
                   seq_len=max_seq_len,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_ffcontroller_curriculum_simple_small_10k_hard_task17():
    """
    Neural Turing machine, associative recall task function.
    """

    batch_size = 150

    # No of hids for controller
    n_hids =  180

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 130

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = False
    use_nogru_mem2q = False
    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7

    max_seq_len = 15
    max_fact_len = 12

    n_read_heads = 2
    n_write_heads = 1
    lambda1_rein = 2e-3
    lambda2_rein = 3e-5
    base_reg = 0.01

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    task_id = 17

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_hard_task17_gru_inp"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=True)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=True,
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
                   seq_len=max_seq_len,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task17():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  128

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 130

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 70

    # Size of the bow embeddings
    bow_size = 128

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    use_quad_interactions = True

    std = 0.05
    seed = 7
    max_seq_len = 4
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 1

    lambda1_rein = 6e-5
    lambda2_rein = 2e-5
    base_reg = 8e-5

    renormalization_scale = 4.0
    w2v_embed_scale = 0.32

    #size of the address in the memory:
    address_size = 24

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)

    NRect = lambda x, use_noise=False: NRect(x,
                                             rng=trng, \
                                             use_noise=use_noise, \
                                             std=std)

    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    task_id = 17

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks_128.pkl"

    l1_pen = 7e-4
    l2_pen = 1e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_hard_nogru2q_task17_nsteps2_gruinp"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen,
                    debug=False)

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

    use_gate_quad_interactions = True
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
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_gru_inp_rep=True,
                   use_gate_quad_interactions=use_gate_quad_interactions,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
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
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task1():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  128

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 130

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 70

    # Size of the bow embeddings
    bow_size = 128

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7
    max_seq_len = 20
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 1

    lambda1_rein = 8e-5
    lambda2_rein = 2e-5
    base_reg = 8e-5

    #size of the address in the memory:
    address_size = 20
    renormalization_scale = None
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    task_id = 1

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks_128.pkl"

    l1_pen = 7e-4
    l2_pen = 1e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_hard_nogru2q_task1"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
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
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()

def proto_ntm_fb_BABI_task_ffcontroller_curriculum_simple_small_10k_hard_task2():
    """
       Neural Turing machine, associative recall task function.
    """
    batch_size = 240

    # No of hids for controller
    n_hids =  128

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 128

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 240

    # Size of the bow embeddings
    bow_size = 200

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.01
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
    renormalization_scale = None
    w2v_embed_scale = 0.05

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = False
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    """
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.2,
                               gamma_clip=0.0)
    """

    learning_rule = Adam(gradient_clipping=10)
    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Sigmoid
    lr = 1e-4

    #w2v_embed_path = False
    w2v_embed_path = None #"new_dict_ngram_false_all_tasks_160.pkl"

    use_reinforce_baseline = False

    l1_pen = 7e-4
    l2_pen = 9e-4
    debug = False

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_1rheads_1whead_l2pen_simple_small_10k_noadv_hard_nogru2q_task2_reinf"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          max_fact_len=tdata_gen.max_fact_len,
                                          max_seq_len=max_seq_len,
                                          randomize=False,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
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
                   use_reinforce=True,
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
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_bowout():
    """
    Neural Turing machine, associative recall task function.
    """

    batch_size = 140

    # No of hids for controller
    n_hids =  120

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 102

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 70

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout = False

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12
    smoothed_diff_weights = False

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 10
    renormalization_scale = 4
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    w2v_embed_path = None
    use_reinforce_baseline = True

    l1_pen = 5e-4
    l2_pen = 9e-4
    debug = False
    anticorr = None

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_small_address"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=bowout)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=bowout,
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
                   anticorrelation=anticorr,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_gru_inp_rep=True,
                   smoothed_diff_weights=smoothed_diff_weights,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
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
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2e-4,
                              bow_weight_start=0.1,
                              bow_weight_stop=0.05,
                              bow_weight_anneal_start=300,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()

def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_bowout_qmask_dbg():
    """
       Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  140

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 102
    smoothed_diff_weights = True

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 70

    # Size of the bow embeddings
    bow_size = 60

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout = True

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 12
    renormalization_scale = 5
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True

    # from theano.compile.debugmode import DebugMode
    # mode = DebugMode(stability_patience=1, check_py_code=False, check_isfinite=False)
    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)
    """
    learning_rule = Adam(gradient_clipping=10)
    """

    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    w2v_embed_path = None
    use_reinforce_baseline = True

    l1_pen = 2e-3
    l2_pen = 8e-4
    debug = False
    anticorr = None

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"

    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v" + \
            "_qmask_small_address_2_sw_rnn_noquad_adam"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=True,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=True,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=bowout, output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=bowout,
                   smoothed_diff_weights=smoothed_diff_weights,
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
                   anticorrelation=anticorr,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=True,
                   use_gru_inp_rep=False,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   correlation_ws=9e-4,
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
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2*1e-4,
                              bow_weight_start=0.74,
                              bow_weight_stop=1.6*1e-1,
                              bow_weight_anneal_start=300,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              use_qmask=True,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()

def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_bowout_qmask():
    """
       Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  140

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 102
    smoothed_diff_weights = True

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 140

    # Size of the bow embeddings
    bow_size = 100

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout = True

    std = 0.03
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 2e-5

    #size of the address in the memory:
    address_size = 12
    renormalization_scale = 5
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)
    """
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    """
    learning_rule = Adam(gradient_clipping=15)

    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 5e-4

    w2v_embed_path = None
    use_reinforce_baseline = True

    l1_pen = 1e-4
    l2_pen = 8.5*1e-4
    debug = True
    anticorr = None

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"

    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v" + \
            "_qmask_small_address_2_sw_avgh0"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=bowout, output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=bowout,
                   smoothed_diff_weights=smoothed_diff_weights,
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
                   anticorrelation=anticorr,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=False,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   correlation_ws=9e-4,
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
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2*1e-4,
                              bow_weight_start=0.68,
                              bow_weight_stop=1e-1,
                              bow_weight_anneal_start=200,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              use_qmask=True,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()

def proto_ntm_fb_BABI_task_grucontroller_simple_small_10k_hard_task2_bowout_qmask_2heads():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 160

    # No of hids for controller
    n_hids =  160

    # No of cols in M
    mem_size = 20

    # No of rows in M
    mem_nel = 102
    smoothed_diff_weights = True

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 160

    # Size of the bow embeddings
    bow_size = 100

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout = True
    anticorr = 6e-4

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 2
    n_write_heads = 1
    n_reading_steps = 1

    lambda1_rein = 2e-5
    lambda2_rein = 1e-5
    base_reg = 1e-5

    #size of the address in the memory:
    address_size = 10
    renormalization_scale = 5.0
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 3e-3

    w2v_embed_path = None
    use_reinforce_baseline = False

    l1_pen = 8e-5
    l2_pen = 4e-3
    debug = False

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"

    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v" + \
            "_qmask_small_address_2_sw_avgh0_grucont"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen,
                    debug=debug,
                    use_bow_out=bowout,
                    output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=bowout,
                   smoothed_diff_weights=smoothed_diff_weights,
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
                   anticorrelation=anticorr,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=False,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   correlation_ws=5e-4,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
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
                              print_every=50,
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=1.6*1e-4,
                              bow_weight_start=0.74,
                              bow_weight_stop=1.4*1e-1,
                              bow_weight_anneal_start=300,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              use_qmask=True,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()



def proto_ntm_fb_BABI_task_grucontroller_simple_small_10k_hard_task1_bowout_qmask():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 160
    sub_mb_size = 160

    # No of hids for controller
    n_hids =  160

    # No of cols in M
    mem_size = 20

    # No of rows in M
    mem_nel = 102
    smoothed_diff_weights = True

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 100

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout = True

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 3e-5
    lambda2_rein = 1e-5
    base_reg = 1e-5

    #size of the address in the memory:
    address_size = 10
    renormalization_scale = 5.0
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adam(gradient_clipping=10)

    task_id = 1

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 3e-3

    w2v_embed_path = None
    use_reinforce_baseline = False

    l1_pen = 1e-4
    l2_pen = 4e-3
    debug = False
    anticorr = None

    path = "/raid/gulcehrc/tasks_1-20_v1-2/en-10k/"

    prfx = "ntm_on_fb_BABI_task_1_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_now2v" + \
            "_qmask_small_address_2_sw_avgh0_grucont"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=False,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen,
                    debug=debug,
                    use_bow_out=bowout,
                    output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=bowout,
                   smoothed_diff_weights=smoothed_diff_weights,
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
                   anticorrelation=anticorr,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=False,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   correlation_ws=3e-3,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
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
                              print_every=50,
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=1.6*1e-4,
                              bow_weight_start=0.74,
                              bow_weight_stop=1.4*1e-1,
                              bow_weight_anneal_start=300,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              use_qmask=True,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_simple_small_10k_hard_task2_bowout_qmask():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 160

    # No of hids for controller
    n_hids =  160

    # No of cols in M
    mem_size = 20

    # No of rows in M
    mem_nel = 102
    smoothed_diff_weights = True

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 80

    # Size of the bow embeddings
    bow_size = 100

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout = True

    std = 0.03
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 1e-5
    lambda2_rein = 7e-6
    base_reg = 1e-5

    #size of the address in the memory:
    address_size = 12
    renormalization_scale = 5.0
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adam(gradient_clipping=15)

    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 3e-3

    w2v_embed_path = None
    use_reinforce_baseline = False

    l1_pen = 8e-5
    l2_pen = 4e-3
    debug = False
    anticorr = None

    path = "/raid/gulcehrc/tasks_1-20_v1-2/en-10k/"

    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v" + \
            "_qmask_small_address_2_sw_avgh0_grucont"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen,
                    debug=debug,
                    use_bow_out=bowout,
                    output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=bowout,
                   smoothed_diff_weights=smoothed_diff_weights,
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
                   anticorrelation=anticorr,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=False,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   correlation_ws=3e-3,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
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
                              print_every=50,
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=1.6*1e-4,
                              bow_weight_start=0.74,
                              bow_weight_stop=1.4*1e-1,
                              bow_weight_anneal_start=300,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              use_qmask=True,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_bowout_qmask_inspectonly():
    """
       Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  160

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 102
    smoothed_diff_weights = True

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 70

    # Size of the bow embeddings
    bow_size = 72

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout = True

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 12
    renormalization_scale = 5
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    """
    learning_rule = Adam(gradient_clipping=10)
    """

    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    w2v_embed_path = None
    use_reinforce_baseline = True

    l1_pen = 2e-3
    l2_pen = 8.5*1e-4
    debug = False
    anticorr = None

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"

    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v" + \
            "_qmask_small_address_2_sw_avgh0"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=bowout, output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=bowout,
                   smoothed_diff_weights=smoothed_diff_weights,
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
                   anticorrelation=anticorr,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=False,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   correlation_ws=9e-4,
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
                              inspect_only=True,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              use_qmask=True,
                              reload_model=True,
                              max_iters=80000,
                              prefix=prfx)

    batch = vdata_gen.next()
    main_loop.inspect_model(batch['x'], batch['mask'], batch['cmask'])


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task3_bowout_qmask():
    """
       Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  160

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 102
    smoothed_diff_weights = True

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 70

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout = True

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 230
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 3

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 12
    renormalization_scale = None
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    """
    learning_rule = Adam(gradient_clipping=10)
    """

    task_id = 3

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    w2v_embed_path = None
    use_reinforce_baseline = True

    l1_pen = 2e-3
    l2_pen = 8e-4
    debug = False
    anticorr = None

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"

    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_3read_steps_quad_interactions" + \
            "_mem_tanh_l2pen_simple_small_10k_hard_nogru2q_task3_loc_addr_now2v" + \
            "_qmask_small_address_sw_avgh0"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=bowout, output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=bowout,
                   smoothed_diff_weights=smoothed_diff_weights,
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
                   anticorrelation=anticorr,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=False,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   correlation_ws=8e-4,
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
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2.5*1e-4,
                              bow_weight_start=0.64,
                              bow_weight_stop=1.2*1e-1,
                              bow_weight_anneal_start=300,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              use_qmask=True,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_anticorr():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  140

    # No of cols in M
    mem_size = 36

    # No of rows in M
    mem_nel = 100

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 140

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    smoothed_diff_weights = True

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 5e-5
    lambda2_rein = 2e-5
    base_reg = 5e-5
    anticorr = None

    # size of the address in the memory:
    address_size = 18
    renormalization_scale = 5
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    task_id = 2
    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    use_context = False
    use_qmask = False

    w2v_embed_path = None
    use_reinforce_baseline = False

    l1_pen = 4e-4
    l2_pen = 2e-4

    debug = False

    path = "/data/lisatmp2/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_acorr_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_adam_noprednext"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    tsdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)


    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=True,
                   mem_size=mem_size,
                   mem_nel=mem_nel,
                   use_ff_controller=use_ff_controller,
                   sub_mb_size=sub_mb_size,
                   deep_out_size=deep_out_size,
                   inps=inps,
                   use_context=use_context,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   smoothed_diff_weights = smoothed_diff_weights,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   anticorrelation=anticorr,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
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
                   use_qmask=use_qmask,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2e-4,
                              bow_weight_stop=0.12,
                              bow_weight_anneal_start=400,
                              bow_weight_start=0.68,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              test_data_gen=tsdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              use_qmask=use_qmask,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_anticorr():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  140

    # No of cols in M
    mem_size = 36

    # No of rows in M
    mem_nel = 100

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 140

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    smoothed_diff_weights = True

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 5e-5
    lambda2_rein = 2e-5
    base_reg = 5e-5
    anticorr = None

    # size of the address in the memory:
    address_size = 18
    renormalization_scale = 5
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    task_id = 2
    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    use_context = False
    use_qmask = False

    w2v_embed_path = None
    use_reinforce_baseline = False

    l1_pen = 4e-4
    l2_pen = 2e-4

    debug = False

    path = "/data/lisatmp2/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_acorr_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_adam_noprednext"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    tsdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)


    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=True,
                   mem_size=mem_size,
                   mem_nel=mem_nel,
                   use_ff_controller=use_ff_controller,
                   sub_mb_size=sub_mb_size,
                   deep_out_size=deep_out_size,
                   inps=inps,
                   use_context=use_context,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   smoothed_diff_weights = smoothed_diff_weights,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   anticorrelation=anticorr,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
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
                   use_qmask=use_qmask,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2e-4,
                              bow_weight_stop=0.12,
                              bow_weight_anneal_start=400,
                              bow_weight_start=0.68,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              test_data_gen=tsdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              use_qmask=use_qmask,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_anticorr():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  140

    # No of cols in M
    mem_size = 36

    # No of rows in M
    mem_nel = 100

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 140

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    smoothed_diff_weights = True

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 5e-5
    lambda2_rein = 2e-5
    base_reg = 5e-5
    anticorr = None

    # size of the address in the memory:
    address_size = 18
    renormalization_scale = 5
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    task_id = 2
    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    use_context = False
    use_qmask = False

    w2v_embed_path = None
    use_reinforce_baseline = False

    l1_pen = 4e-4
    l2_pen = 2e-4

    debug = False

    path = "/data/lisatmp2/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_acorr_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_adam_noprednext"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    tsdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)


    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=True,
                   mem_size=mem_size,
                   mem_nel=mem_nel,
                   use_ff_controller=use_ff_controller,
                   sub_mb_size=sub_mb_size,
                   deep_out_size=deep_out_size,
                   inps=inps,
                   use_context=use_context,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   smoothed_diff_weights = smoothed_diff_weights,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   anticorrelation=anticorr,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
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
                   use_qmask=use_qmask,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2e-4,
                              bow_weight_stop=0.12,
                              bow_weight_anneal_start=400,
                              bow_weight_start=0.68,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              test_data_gen=tsdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              use_qmask=use_qmask,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_anticorr():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  140

    # No of cols in M
    mem_size = 36

    # No of rows in M
    mem_nel = 100

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 140

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    smoothed_diff_weights = True

    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 5e-5
    lambda2_rein = 2e-5
    base_reg = 5e-5
    anticorr = None

    # size of the address in the memory:
    address_size = 18
    renormalization_scale = 5
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    task_id = 2
    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    use_context = False
    use_qmask = False

    w2v_embed_path = None
    use_reinforce_baseline = False

    l1_pen = 4e-4
    l2_pen = 2e-4

    debug = False

    path = "/data/lisatmp2/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_acorr_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_adam_noprednext"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    tsdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)


    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=True,
                   mem_size=mem_size,
                   mem_nel=mem_nel,
                   use_ff_controller=use_ff_controller,
                   sub_mb_size=sub_mb_size,
                   deep_out_size=deep_out_size,
                   inps=inps,
                   use_context=use_context,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   smoothed_diff_weights = smoothed_diff_weights,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   anticorrelation=anticorr,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
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
                   use_qmask=use_qmask,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2e-4,
                              bow_weight_stop=0.12,
                              bow_weight_anneal_start=400,
                              bow_weight_start=0.68,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              test_data_gen=tsdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              use_qmask=use_qmask,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_soft_task1_anticorr():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 160

    # No of hids for controller
    n_hids =  140

    # No of cols in M
    mem_size = 30

    # No of rows in M
    mem_nel = 102

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 160

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    smoothed_diff_weights = True

    std = 0.03
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 10

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 3

    lambda1_rein = 5e-5
    lambda2_rein = 2e-5
    base_reg = 8e-5
    anticorr = None

    # size of the address in the memory:
    address_size = 18
    renormalization_scale = 3.5
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               #grad_clip=0.25,
                               gamma_clip=0.0)

    task_id = 1
    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    use_context = False
    use_qmask = False

    w2v_embed_path = None
    use_reinforce_baseline = False

    l1_pen = 6e-4
    l2_pen = 3e-4

    debug = False

    path = "/data/lisatmp2/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_acorr_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task1_loc_addr_now2v_adam_noprednext_norf"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=False,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    tsdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=False,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)


    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   predict_bow_out=True,
                   mem_size=mem_size,
                   mem_nel=mem_nel,
                   use_ff_controller=use_ff_controller,
                   sub_mb_size=sub_mb_size,
                   deep_out_size=deep_out_size,
                   inps=inps,
                   use_context=use_context,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   smoothed_diff_weights = smoothed_diff_weights,
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   anticorrelation=anticorr,
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
                   use_qmask=use_qmask,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=500,
                              validate_every=500,
                              bow_out_anneal_rate=2e-4,
                              bow_weight_stop=0.12,
                              bow_weight_anneal_start=400,
                              bow_weight_start=0.68,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              test_data_gen=tsdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              use_qmask=use_qmask,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_bowout_inspectonly():
    """
    Neural Turing machine, associative recall task function.
    """

    batch_size = 120

    # No of hids for controller
    n_hids =  160

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 100

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 60

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = False
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout=False
    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 20
    renormalization_scale = 4
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    w2v_embed_path = None
    use_reinforce_baseline = True

    l1_pen = 5e-4
    l2_pen = 9e-4

    debug = False

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_qmask"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=True,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=True,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=bowout, output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
                           center=0.0)

    bi = BiasInitializer(sparsity=-1,
                         scale=std,
                         rng=rng,
                         init_method=BiasInitMethods.Constant,
                         center=0.0)

    print "Length of the vocabulary, ", len(tdata_gen.vocab.items())

    learning_rule = None
    ntm = NTMModel(n_in=len(tdata_gen.vocab.items()),
                   n_hids=n_hids,
                   bow_size=bow_size,
                   n_out=len(tdata_gen.vocab.items()),
                   predict_bow_out=bowout,
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
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
                   correlation_ws=1e-2,
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
                              inspect_only=True,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=True,
                              max_iters=80000,
                              prefix=prfx)

    batch = vdata_gen.next()
    main_loop.inspect_model(batch['x'], batch['mask'], batch['cmask'])


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_bowout_qmask_inspectonly():
    batch_size = 120

    # No of hids for controller
    n_hids =  100

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 100

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 60

    # Size of the bow embeddings
    bow_size = 80

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = False
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False
    bowout=True
    std = 0.025
    seed = 7
    bow_out_reg = 8e-1
    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 2

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 20
    renormalization_scale = 4
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = True
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    w2v_embed_path = None
    use_reinforce_baseline = True

    l1_pen = 5e-4
    l2_pen = 9e-4

    debug = False

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_qmask_pred"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          predict_next_bow=True,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          predict_next_bow=True,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=bowout)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
                           center=0.0)

    bi = BiasInitializer(sparsity=-1,
                         scale=std,
                         rng=rng,
                         init_method=BiasInitMethods.Constant,
                         center=0.0)

    print "Length of the vocabulary, ", len(tdata_gen.vocab.items())

    learning_rule = None
    ntm = NTMModel(n_in=len(tdata_gen.vocab.items()),
                   n_hids=n_hids,
                   bow_size=bow_size,
                   n_out=len(tdata_gen.vocab.items()),
                   predict_bow_out=bowout,
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
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   debug=debug,
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
                              inspect_only=True,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=True,
                              max_iters=80000,
                              prefix=prfx)
    batch = tdata_gen.next()
    main_loop.inspect_model(batch['x'], batch['mask'], batch['cmask'])


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task3_bowout():
    """
    Neural Turing machine, associative recall task function.
    """

    batch_size = 160

    # No of hids for controller
    n_hids =  180

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 160

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 80

    # Size of the bow embeddings
    bow_size = 100

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.04
    seed = 7
    bow_out_reg = 3e-1
    max_seq_len = 250
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 3

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 20
    renormalization_scale = 4
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = False
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)

    """
    learning_rule = Adam(gradient_clipping=10)
    """

    task_id = 3

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 6e-3

    w2v_embed_path = None #"new_dict_ngram_false_all_tasks_160.pkl"
    use_reinforce_baseline = True

    l1_pen = 8e-4
    l2_pen = 6e-4

    debug = False

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task3_loc_addr_now2v_adam"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.AdaptiveUniXav,
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
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   predict_bow_out=False,
                   debug=debug,
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
                              checkpoint_every=500,
                              validate_every=500,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()

def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2():
    """
       Neural Turing machine, associative recall task function.
    """
    batch_size = 128

    # No of hids for controller
    n_hids =  160

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 160

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 64

    # Size of the bow embeddings
    bow_size = 160

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.03
    seed = 7

    max_seq_len = 100
    max_fact_len = 12

    n_read_heads = 2
    n_write_heads = 1
    n_reading_steps = 1

    lambda1_rein = 4e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 20
    renormalization_scale = 4
    w2v_embed_scale = 0.15

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = False
    mode = None
    import sys
    sys.setrecursionlimit(50000)

    """
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.25,
                               gamma_clip=0.0)
    """

    learning_rule = Adam(gradient_clipping=10)
    task_id = 2

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Trect
    lr = 6e-3

    w2v_embed_path = None #"new_dict_ngram_false_all_tasks_160.pkl"
    use_reinforce_baseline = True

    l1_pen = 8e-4
    l2_pen = 6e-4

    debug = False

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_2read_steps_quad_interactions" + \
            "_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_adam"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_test_ngram_False.pkl',
                                          randomize=False,
                                          max_fact_len=tdata_gen.max_fact_len,
                                          max_seq_len=max_seq_len,
                                          task_id=task_id,
                                          task_path=path,
                                          fact_vocab="all_tasks_test_ngram_False_dict.pkl",
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
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   bow_out_reg=bow_out_reg,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
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
                              checkpoint_every=500,
                              validate_every=500,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)
    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task17_v2():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 140

    # No of hids for controller
    n_hids =  128

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 130

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 70

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7
    max_seq_len = 4
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 1

    lambda1_rein = 1e-5
    lambda2_rein = 1e-5
    base_reg = 3e-5

    #size of the address in the memory:
    address_size = 24
    renormalization_scale = 10
    w2v_embed_scale = 0.36
    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    task_id = 17

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 1e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_hard_nogru2q_task17_nsteps2_gruinp_v2"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   use_gru_inp_rep=True,
                   use_bow_input=False,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
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
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_alltasks():
    """
    Neural Turing machine, associative recall task function.
    """
    batch_size = 160

    # No of hids for controller
    n_hids =  128

    # No of cols in M
    mem_size = 20

    # No of rows in M
    mem_nel = 260

    # Not using deep out
    deep_out_size = 100
    sub_mb_size = 80

    # Size of the bow embeddings
    bow_size = 128

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False

    use_quad_interactions = True
    renormalization_scale = None
    w2v_embed_scale = 0.1

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7
    max_seq_len = 340
    max_fact_len = 15

    n_read_heads = 1
    n_write_heads = 1
    n_reading_steps = 1

    lambda1_rein = 8e-5
    lambda2_rein = 1e-5
    base_reg = 8e-5

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    task_id = None

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks_128.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_noadv_hard_nogru2q_alltasks"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          max_seq_limit=max_seq_len-1,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   n_read_heads=n_read_heads,
                   use_gate_quad_interactions=use_quad_interactions,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
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
                   seq_len=max_seq_len,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_soft_task17():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 180

    # No of hids for controller
    n_hids =  180

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 140

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False
    # Use loc based addressing:
    use_loc_based_addressing = True

    std = 0.05
    seed = 7
    max_seq_len = 4
    max_fact_len = 12

    n_read_heads = 2
    n_write_heads = 1
    lambda1_rein = 2e-3
    lambda2_rein = 3e-5
    base_reg = 0.01

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    task_id = 17

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_soft_task17_lba"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)
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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   use_deepout=False,
                   use_reinforce=False,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
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
                   seq_len=max_seq_len,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              inspect_every=800,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()

def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task19():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 160

    # No of hids for controller
    n_hids =  180

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 132

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False
    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7
    max_seq_len = 10
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    lambda1_rein = 2e-4
    lambda2_rein = 2e-5
    n_reading_steps = 2

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    task_id = 19

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_" + \
            "noadv_hard_nogru2q_task19_v2"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
                   controller_activ=cont_act,
                   use_adv_indexing=False,
                   use_out_mem=False,
                   n_reading_steps=n_reading_steps,
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
                   seq_len=max_seq_len,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task18():
    """
       Neural Turing machine, associative recall task function.
    """
    batch_size = 160

    # No of hids for controller
    n_hids =  200

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 132

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 120

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False
    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7
    max_seq_len = 23
    max_fact_len = 12

    n_read_heads = 1
    n_write_heads = 1
    lambda1_rein = 1.5*1e-3
    lambda2_rein = 3e-5

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    task_id = 18

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7

    #w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"
    w2v_embed_path = "new_dict_ngram_false_all_tasks_120.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_hard_nogru2q_task18"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=True)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
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
                   seq_len=max_seq_len,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task7():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 150

    # No of hids for controller
    n_hids =  180

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 132

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False
    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7
    max_seq_len = 54
    max_fact_len = 8

    n_read_heads = 2
    n_write_heads = 1
    lambda1_rein = 2e-4
    lambda2_rein = 2e-5

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    task_id = 7

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_hard_nogru2q_task7"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=True)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
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
                   seq_len=max_seq_len,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()

def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 150

    # No of hids for controller
    n_hids =  180

    # No of cols in M
    mem_size = 28

    # No of rows in M
    mem_nel = 132

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True
    use_nogru_mem2q = False
    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7
    max_seq_len = 30
    max_fact_len = 8

    n_read_heads = 1
    n_write_heads = 1
    lambda1_rein = 1e-4
    lambda2_rein = 2e-5

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    task_id = 6

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_hard_nogru2q_task6"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = max_seq_len,
                                          max_fact_len = max_fact_len,
                                          task_id = task_id,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = max_seq_len,
                                          randomize = False,
                                          task_id = task_id,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   use_deepout=False,
                   use_reinforce=True,
                   use_nogru_mem2q=use_nogru_mem2q,
                   use_reinforce_baseline=False,
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
                   seq_len=max_seq_len,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_ffcontroller_curriculum_simple_small_10k_hard_insp():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 140

    # No of hids for controller
    n_hids =  100

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 128

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_ffcont_hard_rav"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = 20,
                                          max_fact_len = 7,
                                          task_id = 1,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = 20,
                                          randomize = False,
                                          task_id = 1,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   use_reinforce=True,
                   use_reinforce_baseline=True,
                   controller_activ=cont_act,
                   use_adv_indexing=False,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   seq_len=20,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              inspect_only=True,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=True,
                              max_iters=80000,
                              prefix=prfx)

    batch = vdata_gen.next()
    main_loop.inspect_model(batch['x'], batch['mask'], batch['cmask'])


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_insp():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 140

    # No of hids for controller
    n_hids =  100

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 128

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    cont_act = Tanh
    mem_gater_activ = Trect
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    w2v_embed_path = "new_dict_ngram_false_all_tasks.pkl"

    l1_pen = 7e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_hard"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = 20,
                                          max_fact_len = 7,
                                          task_id = 1,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = 20,
                                          randomize = False,
                                          task_id = 1,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   w2v_embed_path=w2v_embed_path,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   use_reinforce=True,
                   use_reinforce_baseline=False,
                   controller_activ=cont_act,
                   use_adv_indexing=False,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   seq_len=20,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              inspect_only=True,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=True,
                              max_iters=80000,
                              prefix=prfx)

    batch = vdata_gen.next()
    main_loop.inspect_model(batch['x'], batch['mask'], batch['cmask'])


def proto_ntm_fb_BABI_task_ffcontroller_curriculum_simple_small_10k_insp():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 140

    # No of hids for controller
    n_hids =  140

    # No of cols in M
    mem_size = 24

    # No of rows in M
    mem_nel = 128

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = False

    # Use loc based addressing:
    use_loc_based_addressing = False

    std = 0.05
    seed = 7

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7
    task_id = 20

    l1_pen = 6e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_2rheads_1whead_tiny_l2pen_simple_small_10k_noadv_ffcont_hard_rav_t20_best"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = True,
                                          max_seq_len = 15,
                                          max_fact_len = 8,
                                          task_id = 20,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = 15,
                                          randomize = False,
                                          task_id = 20,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   use_reinforce=True,
                   use_reinforce_baseline=True,
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
                   seq_len=15,
                   max_fact_len=tdata_gen.max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=True,
                              inspect_only=True,
                              max_iters=80000,
                              prefix=prfx)

    batch = vdata_gen.next()
    main_loop.inspect_model(batch['x'], batch['mask'], batch['cmask'])


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_rec_insp():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 144

    # No of hids for controller
    n_hids =  90

    # No of cols in M
    mem_size = 32

    # No of rows in M
    mem_nel = 40

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True

    # Use loc based addressing:
    use_loc_based_addressing = True

    std = 0.05
    seed = 7

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 10

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7

    l1_pen = 6e-4
    l2_pen = 2e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_3rheads_1whead_tiny_l2pen_simple_small_10k_noadv_best"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = False,
                                          max_seq_len = 20,
                                          max_fact_len = 7,
                                          task_id = 1,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = 20,
                                          randomize = False,
                                          task_id = 1,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   controller_activ=cont_act,
                   use_adv_indexing=False,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   seq_len=20,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=True,
                              inspect_only=True,
                              max_iters=80000,
                              prefix=prfx)
    batch = vdata_gen.next()
    main_loop.inspect_model(batch['x'], batch['mask'], batch['cmask'])


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_1k():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 32

    # No of hids for controller
    n_hids =  100

    #No of cols in M
    mem_size = 30

    #No of rows in M
    mem_nel = 200

    #Not using deep out
    deep_out_size = 100
    use_deepout = False

    # Size of the bow embeddings
    bow_size = 60

    # ff controller
    use_ff_controller = False

    # For RNN controller:
    learn_h0 = True

    # Use loc based addressing:
    use_loc_based_addressing = True

    std = 0.05
    seed = 7

    n_read_heads = 2
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = True

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    cont_act = Rect
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-7

    l1_pen = 1e-4
    l2_pen = 1e-5

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_3rheads_1whead_tiny_l2pen_simple_small_1k"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                             randomize = False,
                                             max_seq_len = 20,
                                             max_fact_len = 7,
                                             task_id = 1,
                                             task_path = path,
                                             mode='train',
                                             fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                             batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                             max_fact_len = tdata_gen.max_fact_len,
                                             max_seq_len = 20,
                                             randomize = False,
                                             task_id = 1,
                                             task_path = path,
                                             fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                             batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   controller_activ=cont_act,
                   use_adv_indexing=True,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   train_profile=True,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   seq_len=20,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


def proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k():
    """
       Neural Turing machine, associative recall task function.
    """

    batch_size = 128

    # No of hids for controller
    n_hids =  100

    #No of cols in M
    mem_size = 20

    #No of rows in M
    mem_nel = 200

    #Not using deep out
    deep_out_size = 100
    use_deepout = False

    # Size of the bow embeddings
    bow_size = 64

    # ff controller
    use_ff_controller = True

    # For RNN controller:
    learn_h0 = False

    # Use loc based addressing:
    use_loc_based_addressing = True

    std = 0.05
    seed = 7

    n_read_heads = 1
    n_write_heads = 1

    #size of the address in the memory:
    address_size = 20

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = True

    mode = None
    import sys
    sys.setrecursionlimit(50000)
    learning_rule = Adasecant2(delta_clip=25,
                               use_adagrad=True,
                               grad_clip=0.5,
                               gamma_clip=0.0)
    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    lr = 1e-3

    l1_pen = 1e-4
    l2_pen = 1e-4

    path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/"
    prfx = "ntm_on_fb_BABI_task_all_addr10_learn_h0_l1_no_" + \
            "out_mem_lin_start_tanh_curr_3rheads_1whead_tiny_l2pen_simple_small_10k"

    tdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_train_ngram_False.pkl',
                                          randomize = False,
                                          max_seq_len = 20,
                                          max_fact_len = 7,
                                          task_id = 1,
                                          task_path = path,
                                          mode='train',
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file = 'all_tasks_test_ngram_False.pkl',
                                          randomize = False,
                                          max_fact_len = tdata_gen.max_fact_len,
                                          max_seq_len = 20,
                                          task_id = 1,
                                          task_path = path,
                                          fact_vocab = "all_tasks_test_ngram_False_dict.pkl",
                                          batch_size = batch_size)

    inps = get_inps(vgen=vdata_gen, debug=False)

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
                   deep_out_size=deep_out_size,
                   inps=inps,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   erase_activ=erase_activ,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   learning_rule=learning_rule,
                   use_deepout=False,
                   controller_activ=cont_act,
                   use_adv_indexing=True,
                   use_out_mem=False,
                   unroll_recurrence=False,
                   address_size=address_size,
                   learn_h0=learn_h0,
                   theano_function_mode=mode,
                   l1_pen=l1_pen,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   seq_len=20,
                   softmax=True,
                   batch_size=batch_size)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=40,
                              checkpoint_every=400,
                              validate_every=400,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              learning_rate=lr,
                              reload_model=False,
                              valid_iters=None,
                              linear_start=False,
                              max_iters=80000,
                              prefix=prfx)

    main_loop.run()


