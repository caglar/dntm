import theano
import theano.tensor as TT

import numpy as np
import sys
sys.path.append("../codes/")

from core.learning_rule import Adam
from core.parameters import (WeightInitializer,
                                    BiasInitializer,
                                    InitMethods,
                                    BiasInitMethods)

from core.nan_guard import NanGuardMode

from core.commons import Tanh, Trect, Sigmoid, Rect, Leaky_Rect
from mainloop import SeqMNISTMainLoop
from nmodel import NTMModel

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pprint as pp

import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes


np.random.seed(123)


_datasets = None
def get_dataset(which_set):
    global _datasets
    if not _datasets:
        MNIST = fuel.datasets.MNIST
        # jump through hoops to instantiate only once and only if needed
        _datasets = dict(
            train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
            test=MNIST(which_sets=["test"]))
    return _datasets[which_set]


def get_stream(which_set, batch_size, num_examples=None):
    dataset = get_dataset(which_set)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream


def search_model_adam(state, channel, reload_model=False):

    pp.pprint(state)

    def NReLU(x, rng=None, use_noise=False):
        assert rng is not None
        if use_noise:
            stds = Sigmoid(x)
            x = x + rng.normal(x.shape, avg=0.0, std=stds, dtype=x.dtype)
        return Trect(x)

    def get_inps(vgen=None, debug=False, output_map=None):

        X, y = TT.fmatrix("X"), TT.vector("y", dtype="uint8")

        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.get_epoch_iterator().next()
            X.tag.test_value = batch[0].reshape((batch[0].shape[0], -1))
            y.tag.test_value = batch[1].flatten()
        return [X, y]

    lr = state['lr']
    batch_size = state['batch_size']

    # No of els in the cols of the content for the memory
    mem_size = state['mem_size']

    # No of rows in M
    mem_nel = state['mem_nel']
    std = state['std']
    renormalization_scale = state['renormalization_scale']
    sub_mb_size = state['sub_mb_size']
    smoothed_diff_weights = state.get('smoothed_diff_weights', False)

    max_len = 784
    inp_size = 1

    # No of hids for controller
    n_hids = state['n_hids']

    # Not using deep out
    deep_out_size = 100

    # Size of the bow embeddings
    bow_size = state.get('bow_size', 80)

    # ff controller
    use_ff_controller = state['use_ff_controller']

    # For RNN controller:
    learn_h0 = state.get('learn_h0', False)
    use_nogru_mem2q = False

    # Use loc based addressing:
    use_loc_based_addressing = state.get('use_loc_based_addressing', False)
    bowout = state.get('bowout', False)
    use_reinforce = state.get('use_reinforce', False)
    permute_order = state.get('permute_order', True)

    seed = 7
    n_read_heads = state['n_read_heads']
    n_write_heads = 1
    n_reading_steps = state['n_reading_steps']

    lambda1_rein = state.get('lambda1_rein', 4e-5)
    lambda2_rein = state.get('lambda2_rein', 1e-5)
    base_reg = 2e-5

    #size of the address in the memory:
    address_size = state["address_size"]
    w2v_embed_scale = 0.05
    n_out = 10

    rng = np.random.RandomState(seed)
    trng = RandomStreams(seed)
    NRect = lambda x, use_noise=False: NRect(x, rng=trng, use_noise=use_noise, std=std)
    use_noise = False

    use_quad_interactions = state.get('use_quad_interactions', True)

    mode = state.get('theano_function_mode', None)
    import sys
    sys.setrecursionlimit(50000)

    learning_rule = Adam(gradient_clipping=state.get('gradient_clip', 10))

    cont_act = Tanh
    mem_gater_activ = Sigmoid
    erase_activ = Sigmoid
    content_activ = Tanh
    use_gru_inp = state.get('use_gru_inp', False)
    use_bow_inp = state.get('use_bow_inp', False)

    w2v_embed_path = None
    use_reinforce_baseline = state['use_reinforce_baseline']

    use_reinforce = state.get('use_reinforce', False)
    l1_pen = state.get('l1_pen', 1e-4)
    l2_pen = state.get('l2_pen', 1e-3)
    hybrid_att = state.get('hybrid_att', False)
    use_dice_val = state.get('use_dice_val', False)
    debug = state.get('debug', False)
    correlation_ws = state.get('correlation_ws', False)

    idxs = np.arange(max_len)
    np.random.shuffle(idxs)
    use_batch_norm = state.get("use_batch_norm", False)
    anticorr = state.get('anticorr', None)
    prfx = ("ntm_on_fb_copy_task_all_learn_h0_l1_no_n_hids_%(n_hids)s_bsize_%(batch_size)d"
            "_std_%(std)f_mem_nel_%(mem_nel)d_mem_size_%(mem_size)f_lr_%(lr)f_use_bn_%(use_batch_norm)d_hard2") % locals()

    tdata_gen = get_stream(which_set="train",
                           batch_size=batch_size)

    vdata_gen = get_stream(which_set="valid",
                           batch_size=batch_size)

    tst_data_gen = get_stream(which_set="test",
                              batch_size=batch_size)

    n_layers = state.get('n_layers', 1)
    inps = get_inps(vgen=vdata_gen,
                    debug=debug,
                    output_map=True)

    wi = WeightInitializer(sparsity=-1,
                           scale=std,
                           rng=rng,
                           init_method=InitMethods.Adaptive,
                           center=0.0)

    bi = BiasInitializer(sparsity=-1,
                         scale=1e-3,
                         rng=rng,
                         init_method=BiasInitMethods.Random,
                         center=0.0)

    ntm = NTMModel(n_in=inp_size,
                   n_hids=n_hids,
                   bow_size=bow_size,
                   n_out=n_out,
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
                   use_batch_norm=use_batch_norm,
                   w2v_embed_scale=w2v_embed_scale,
                   n_read_heads=n_read_heads,
                   n_write_heads=n_write_heads,
                   use_last_hidden_state=True,
                   use_loc_based_addressing=use_loc_based_addressing,
                   use_simple_rnn_inp_rep=False,
                   use_gru_inp_rep=use_gru_inp,
                   use_bow_input=use_bow_inp,
                   use_inp_content=False,
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
                   use_cost_mask=False,
                   use_noise=use_noise,
                   rnd_indxs=idxs,
                   permute_order=permute_order,
                   max_fact_len=max_len,
                   softmax=True,
                   batch_size=None)

    save_freq = state.get("save_freq", 1000)

    main_loop = SeqMNISTMainLoop(ntm,
                                 print_every=50,
                                 checkpoint_every=save_freq,
                                 validate_every=500,
                                 train_data_gen=tdata_gen,
                                 valid_data_gen=vdata_gen,
                                 test_data_gen=tst_data_gen,
                                 learning_rate=lr,
                                 reload_model=reload_model,
                                 num_epochs=250,
                                 state=state,
                                 prefix=prfx)
    main_loop.run()


if __name__=="__main__":

    from collections import OrderedDict
    state = OrderedDict({})

    state.lr = 1e-3
    state.batch_size = 128
    state['sub_mb_size']= None
    state.std = 0.05
    state.max_iters = 80000
    state.n_hids = 100
    state.mem_size = 8
    state.mem_nel = 100
    state['n_reading_steps'] = 1
    state['n_read_heads'] = 1
    state['address_size'] = 8
    state['use_batch_norm'] = False

    state['use_reinforce_baseline'] = False
    state['use_reinforce'] = False
    state['renormalization_scale'] = None
    state['use_quad_interactions'] = False

    state['l1_pen'] = 7e-6
    state['l2_pen'] = 3e-5
    state['learn_h0'] = True

    state['use_ff_controller'] = False
    state['correlation_ws'] = None
    state['debug'] = False

    search_model_adam(state, channel=None)
