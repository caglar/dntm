import math
import os

# os.environ['THEANO_FLAGS'] += ',base_compiledir=%s/'%(os.environ['LSCRATCH'])
# os.environ['THEANO_FLAGS'] += ',base_compiledir=""'

import theano
import theano.tensor as TT

import numpy as np
import sys
sys.path.append("../codes/")

from core.learning_rule import Adasecant, Adam, RMSPropMomentum, Adasecant2
from core.parameters import (WeightInitializer,
                                    BiasInitializer,
                                    InitMethods,
                                    BiasInitMethods)

from core.nan_guard import NanGuardMode

from core.commons import Tanh, Trect, Sigmoid, Rect, Leaky_Rect
from memnet.mainloop import FBaBIMainLoop
from memnet.nmodel import NTMModel
from memnet.fbABIdataiterator import FBbABIDataIteratorSingleQ

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pprint as pp


def search_model_adam(state, channel, reload_model=False):

    pp.pprint(state)

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

    def get_inps(use_mask=True, vgen=None, use_bow_out=False, debug=False, output_map=None):
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
            return outs
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

    # No of els in the cols of the content for the memory
    mem_size = state.mem_size

    # No of rows in M
    mem_nel = state.mem_nel
    std = state.std
    renormalization_scale = state.renormalization_scale
    sub_mb_size = state.sub_mb_size
    smoothed_diff_weights = state.get('smoothed_diff_weights', True)

    # No of hids for controller
    n_hids =  state.n_hids

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = state.get('bow_size', 80)

    # ff controller
    use_ff_controller = state.use_ff_controller

    # For RNN controller:
    learn_h0 = state.get('learn_h0', False)
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = state.get('use_loc_based_addressing', False)
    bowout = state.get('bowout', True)
    use_reinforce = state.get('use_reinforce', False)

    seed = 7
    max_seq_len = state.max_seq_len
    max_fact_len = state.max_fact_len

    n_read_heads = state.n_read_heads
    n_write_heads = 1
    n_reading_steps = state.n_reading_steps

    lambda1_rein = state.get('lambda1_rein', 4e-5)
    lambda2_rein = state.get('lambda2_rein', 1e-5)
    base_reg = 2e-5

    #size of the address in the memory:
    address_size = state.address_size
    renormalization_scale = state.renormalization_scale
    w2v_embed_scale = 0.05

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = state.get('use_quad_interactions', True)

    mode = state.get('theano_function_mode', None)
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adam(gradient_clipping=state.get('gradient_clip', 10))
    task_id = state.task_id

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    use_gru_inp = state.get('use_gru_inp', True)
    use_bow_inp = state.get('use_bow_inp', False)

    w2v_embed_path = None
    use_reinforce_baseline = state.use_reinforce_baseline
    use_reinforce = state.get('use_reinforce', False)
    l1_pen = state.get('l1_pen', 1e-4)
    l2_pen = state.get('l2_pen', 1e-3)
    hybrid_att = state.get('hybrid_att', False)
    use_dice_val = state.get('use_dice_val', False)
    debug = state.get('debug', False)
    correlation_ws = state.get('correlation_ws', 6e-4)
    anticorr = state.get('anticorr', None)
    path = state.path
    prfx = ("ntm_on_fb_BABI_task_all__learn_h0_l1_no_n_hids_%(n_hids)s_bsize_%(batch_size)d"
            "_std_%(std)f_mem_nel_%(mem_nel)d_mem_size_%(mem_size)f_lr_%(lr)f") % locals()

    prfx = state.save_path+prfx
    tdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_train_ngram_False.pkl',
                                          randomize=True,
                                          max_seq_len=max_seq_len,
                                          max_fact_len=max_fact_len,
                                          task_id=task_id,
                                          task_path=path,
                                          mode='train',
                                          fact_vocab="../all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)

    vdata_gen = FBbABIDataIteratorSingleQ(task_file='all_tasks_valid_ngram_False.pkl',
                                          max_fact_len=tdata_gen.max_fact_len,
                                          max_seq_len=max_seq_len,
                                          randomize=False,
                                          task_id=task_id,
                                          mode="valid",
                                          task_path=path,
                                          fact_vocab="../all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)



    tst_data_gen = FBbABIDataIteratorSingleQ(task_file='../all_tasks_test_ngram_False.pkl',
                                          max_fact_len=tdata_gen.max_fact_len,
                                          max_seq_len=max_seq_len,
                                          randomize=False,
                                          task_id=task_id,
                                          mode="valid",
                                          task_path=path,
                                          fact_vocab="../all_tasks_test_ngram_False_dict.pkl",
                                          batch_size=batch_size)



    n_layers = state.get('n_layers', 1)
    inps = get_inps(vgen=vdata_gen, debug=debug, use_bow_out=bowout, output_map=True)

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
                   predict_bow_out=bowout,
                   mem_size=mem_size,
                   mem_nel=mem_nel,
                   use_ff_controller=use_ff_controller,
                   sub_mb_size=sub_mb_size,
                   deep_out_size=deep_out_size,
                   inps=inps,
                   n_layers=n_layers,
                   hybrid_att=hybrid_att,
                   smoothed_diff_weights=smoothed_diff_weights,
                   baseline_reg=base_reg,
                   w2v_embed_path=w2v_embed_path,
                   renormalization_scale=renormalization_scale,
                   w2v_embed_scale=w2v_embed_scale,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=False,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=False,
                   use_gru_inp_rep=use_gru_inp,
                   use_bow_input=use_bow_inp,
                   anticorr=anticorr,
                   erase_activ=erase_activ,
                   use_gate_quad_interactions=use_quad_interactions,
                   content_activ=content_activ,
                   use_multiscale_shifts=True,
                   correlation_ws=correlation_ws,
                   learning_rule=learning_rule,
                   lambda1_rein=lambda1_rein,
                   lambda2_rein=lambda2_rein,
                   n_reading_steps=n_reading_steps,
                   use_deepout=False,
                   use_reinforce=use_reinforce,
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
                   debug=debug,
                   mem_gater_activ=mem_gater_activ,
                   tie_read_write_gates=False,
                   weight_initializer=wi,
                   bias_initializer=bi,
                   use_cost_mask=True,
                   use_noise=use_noise,
                   max_fact_len=max_fact_len,
                   softmax=True,
                   batch_size=batch_size)

    bow_weight_stop = state.get('bow_weight_stop', 1.2*1e-1)
    bow_weight_anneal_start = state.get('bow_weight_anneal_start', 320)
    bow_weight_start = state.get("bow_weight_start",0.74)
    bow_out_anneal_rate = state.get("bow_out_anneal_rate",2*1e-4)
    save_freq = state.get("save_freq", 1000)

    main_loop = FBaBIMainLoop(ntm,
                              print_every=50,
                              checkpoint_every=save_freq,
                              validate_every=500,
                              bow_out_anneal_rate=bow_out_anneal_rate,
                              bow_weight_start=bow_weight_start,
                              bow_weight_stop=bow_weight_stop,
                              bow_weight_anneal_start=bow_weight_anneal_start,
                              train_data_gen=tdata_gen,
                              valid_data_gen=vdata_gen,
                              test_data_gen=tst_data_gen,
                              learning_rate=lr,
                              reload_model=reload_model,
                              valid_iters=None,
                              linear_start=False,
                              use_qmask=True,
                              max_iters=state.max_iters,
                              state=state,
                              prefix=prfx)
    main_loop.run()

    if channel is None:
        return None
    return channel.COMPLETE
