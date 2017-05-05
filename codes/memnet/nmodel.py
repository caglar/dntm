import logging
from collections import OrderedDict
import cPickle as pkl
import warnings

import numpy as np

import theano
import theano.tensor as TT

from ntm_layers import NTM, NTMFFController
from core.parameters import (WeightInitializer,
                             BiasInitializer)

from core.layers import (AffineLayer,
                         ForkLayer,
                         BOWLayer,
                         MergeLayer,
                         GRULayer,
                         BatchNormLayer,
                         RNNLayer)

from core.parameters import Parameters
from core.basic import Model
from core.costs import kl, nll, huber_loss
from core.utils import safe_grad, global_rng, block_gradient, as_floatX, \
        safe_izip, sharedX

from core.commons import Leaky_Rect, Sigmoid, Rect, Softmax, Tanh
from core.timer import Timer
from core.operators import Dropout, REINFORCE, REINFORCEBaselineExt
from core.penalty import L2Penalty, ReinforcePenalty, AntiCorrelationConstraint, \
                                CorrelationConstraint

from core.training import MinibatchGradPartitioner

logger = logging.getLogger(__name__)
logger.disabled = False


class NTMModel(Model):
    """
    NTM model.
    """
    def __init__(self,
                 n_in,
                 n_hids,
                 n_out,
                 mem_size,
                 mem_nel,
                 deep_out_size,
                 bow_size=40,
                 inps=None,
                 dropout=None,
                 predict_bow_out=False,
                 seq_len=None,
                 n_read_heads=1,
                 n_layers=1,
                 n_write_heads=1,
                 train_profile=False,
                 erase_activ=None,
                 content_activ=None,
                 l1_pen=None,
                 l2_pen=None,
                 use_reinforce=False,
                 use_reinforce_baseline=False,
                 n_reading_steps=2,
                 use_gru_inp_rep=False,
                 use_simple_rnn_inp_rep=False,
                 use_nogru_mem2q=False,
                 sub_mb_size=40,
                 lambda1_rein=2e-4,
                 lambda2_rein=2e-5,
                 baseline_reg=1e-2,
                 anticorrelation=None,
                 use_layer_norm=False,
                 recurrent_dropout_prob=-1,
                 correlation_ws=None,
                 hybrid_att=True,
                 max_fact_len=7,
                 use_dice_val=False,
                 use_qmask=False,
                 renormalization_scale=4.8,
                 w2v_embed_scale=0.42,
                 emb_scale=0.32,
                 use_soft_att=False,
                 use_hard_att_eval=False,
                 use_batch_norm=False,
                 learning_rule=None,
                 use_loc_based_addressing=True,
                 smoothed_diff_weights=False,
                 use_multiscale_shifts=True,
                 use_ff_controller=False,
                 use_gate_quad_interactions=False,
                 permute_order=False,
                 wpenalty=None,
                 noise=None,
                 w2v_embed_path=None,
                 glove_embed_path=None,
                 learn_embeds=True,
                 use_last_hidden_state=False,
                 use_adv_indexing=False,
                 use_bow_input=True,
                 use_out_mem=True,
                 use_deepout=True,
                 use_q_mask=False,
                 use_inp_content=True,
                 rnd_indxs=None,
                 address_size=0,
                 learn_h0=False,
                 use_context=False,
                 debug=False,
                 controller_activ=None,
                 mem_gater_activ=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 use_cost_mask=True,
                 use_bow_cost_mask=True,
                 theano_function_mode=None,
                 batch_size=32,
                 use_noise=False,
                 reinforce_decay=0.9,
                 softmax=False,
                 use_mask=False,
                 name="ntm_model",
                 **kwargs):

        assert deep_out_size is not None, ("Size of the deep output "
                                           " should not be None.")

        if sub_mb_size is None:
            sub_mb_size = batch_size

        assert sub_mb_size <= batch_size, "batch_size should be greater than sub_mb_size"
        self.hybrid_att = hybrid_att

        self.state = locals()
        self.use_context = use_context
        self.eps = 1e-8
        self.use_mask = use_mask
        self.l1_pen = l1_pen
        self.l2_pen = l2_pen
        self.l2_penalizer = None
        self.emb_scale = emb_scale
        self.w2v_embed_path = w2v_embed_path
        self.glove_embed_path = glove_embed_path
        self.learn_embeds = learn_embeds
        self.exclude_params = {}

        self.use_gate_quad_interactions = use_gate_quad_interactions
        self.reinforce_decay = reinforce_decay
        self.max_fact_len = max_fact_len
        self.lambda1_reinf = lambda1_rein
        self.lambda2_reinf = lambda2_rein
        self.use_reinforce_baseline = use_reinforce_baseline
        self.use_reinforce = use_reinforce
        self.use_gru_inp_rep = use_gru_inp_rep
        self.use_simple_rnn_inp_rep = use_simple_rnn_inp_rep
        self.use_q_mask = use_q_mask
        self.use_inp_content = use_inp_content
        self.rnd_indxs = rnd_indxs

        self.use_layer_norm = use_layer_norm
        self.recurrent_dropout_prob = recurrent_dropout_prob

        self.n_reading_steps = n_reading_steps
        self.sub_mb_size = sub_mb_size
        self.predict_bow_out = predict_bow_out
        self.correlation_ws = correlation_ws
        self.smoothed_diff_weights = smoothed_diff_weights
        self.use_soft_att = use_soft_att
        self.use_hard_att_eval = use_hard_att_eval

        if anticorrelation and n_read_heads < 2:
            raise ValueError("Anti-correlation of the attention weight"
                              " do not support the multiple read heads.")


        self.anticorrelation = anticorrelation

        if self.predict_bow_out:
            if len(inps) <= 4:
                raise ValueError("The number of inputs should be greater than 4.")

        if l2_pen:
            self.l2_penalizer = L2Penalty(self.l2_pen)

        #assert use_bow_input ^ use_gru_inp_rep ^ self.use_simple_rnn_inp_rep, \
        #        "You should either use GRU or BOW input."

        self.renormalization_scale = renormalization_scale
        self.w2v_embed_scale = w2v_embed_scale

        self.baseline_reg = baseline_reg
        self.inps = inps
        self.erase_activ = erase_activ
        self.use_ff_controller = use_ff_controller
        self.content_activ = content_activ
        self.use_bow_cost_mask = use_bow_cost_mask
        self.ntm_outs = None
        self.theano_function_mode = theano_function_mode
        self.n_in = n_in
        self.dropout = dropout
        self.wpenalty = wpenalty
        self.noise = noise
        self.bow_size = bow_size
        self.use_last_hidden_state = use_last_hidden_state
        self.use_loc_based_addressing = use_loc_based_addressing
        self.train_profile = train_profile
        self.use_nogru_mem2q = use_nogru_mem2q
        self.use_qmask = use_qmask
        self.permute_order = permute_order
        self.use_batch_norm = use_batch_norm

        # Use this if you have a ff-controller because otherwise this is not effective:
        self.n_layers = n_layers
        if self.use_reinforce:
            reinforceCls = REINFORCE
            if not self.use_reinforce_baseline:
                reinforceCls = REINFORCEBaselineExt

            self.Reinforce = reinforceCls(lambda1_reg=self.lambda1_reinf,
                                          lambda2_reg=self.lambda2_reinf,
                                          decay=self.reinforce_decay)

            self.ReaderReinforce = \
                    ReinforcePenalty(reinf_level=self.lambda1_reinf,
                                     maxent_level=self.lambda2_reinf,
                                     use_reinforce_baseline=self.use_reinforce_baseline)
        self.dice_val = None

        if use_dice_val:
            self.dice_val = sharedX(1.)

        self.use_dice_val = use_dice_val
        if bow_size is None:
            raise ValueError("bow_size should be specified.")

        if name is None:
            raise ValueError("name should not be empty.")

        self.n_hids = n_hids
        self.mem_size = mem_size
        self.use_deepout = use_deepout
        self.mem_nel = mem_nel
        self.n_out = n_out
        self.use_out_mem = use_out_mem
        self.use_multiscale_shifts = use_multiscale_shifts
        self.address_size = address_size
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.learn_h0 = learn_h0
        self.use_adv_indexing = use_adv_indexing
        self.softmax = softmax
        self.use_bow_input = use_bow_input
        self.use_cost_mask = use_cost_mask
        self.deep_out_size = deep_out_size
        self.controller_activ = controller_activ
        self.mem_gater_activ = mem_gater_activ
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = inps[0].shape[1]

        #assert self.batch_size >= self.sub_mb_size, ("Minibatch size should be "
        #                                             " greater than the sub minibatch size")
        self.comp_grad_fn = None
        self.name = name
        self.use_noise = use_noise
        self.train_timer = Timer("Training function")
        self.gradfn_timer = Timer("Gradient function")
        self.grads_timer = Timer("Computing the grads")
        self.reset()

        self.seq_len = TT.iscalar('seq_len')
        self.__convert_inps_to_list()

        if debug:
            if self.use_gru_inp_rep or self.use_bow_input:
                self.seq_len.tag.test_value = self.inps[0].tag.test_value.shape[1]
            else:
                self.seq_len.tag.test_value = self.inps[0].tag.test_value.shape[0]

        self.learning_rule = learning_rule
        if self.predict_bow_out:
            self.bow_out_w = TT.fscalar("bow_out_w")
            if debug:
                self.bow_out_w.tag.test_value = np.float32(1.0)
        else:
            self.bow_out_w = 0

    def __convert_inps_to_list(self):
        if isinstance(self.inps, list):
            X = self.inps[0]
            y = self.inps[1]
            if self.use_mask:
                mask = self.inps[2]
                cmask = None
            inps = [X, y]

            if self.use_mask:
                inps += [mask]

            if self.use_cost_mask:
                cmask = self.inps[3]
                inps += [cmask]

            if self.correlation_ws or self.use_qmask:
                self.qmask = self.inps[5]
                inps += [self.qmask]

            if self.predict_bow_out:
                bow_out = self.inps[4]
                inps += [bow_out]

            self.inps = inps
        else:
            X = self.inps['X']
            y = self.inps['y']
            mask = self.inps['mask']
            cmask = None
            inps = [X, y]

            if self.use_mask:
                inps += [mask]

            if self.use_cost_mask:
                cmask = self.inps['cmask']
                inps += [cmask]

            if self.correlation_ws or self.use_qmask:
                self.qmask = self.inps['qmask']
                inps += [self.qmask]

            if self.predict_bow_out:
                bow_out = self.inps['bow_out']
                inps += [bow_out]

        self.inps = inps

    def reset(self):
        self.params = Parameters()
        if self.w2v_embed_path and (self.use_bow_input or self.use_gru_inp_rep):
            self.w2v_embeds = pkl.load(open(self.w2v_embed_path, "rb"))

        if self.glove_embed_path:
            logger.info("Loading the GLOVE embeddings...")
            self.glove_embeds = pkl.load(open(self.glove_embed_path, "rb"))

        self.reg = 0
        self.ntm = None
        self.merge_layer = None
        self.out_layer = None
        self.bow_layer = None
        self.baseline_out = None
        self.bow_pred_out = None

        self.gru_fact_layer_inps = None
        self.gru_fact_layer = None

        self.rnn_fact_layer_inps = None
        self.rnn_fact_layer = None

        self.bow_out_layer = None

        self.inp_proj_layer = None
        self.batch_norm_layer = None

        self.children = []
        self.trainpartitioner = None
        self.known_grads = OrderedDict({})
        self.updates = OrderedDict({})

    def __init_to_embeds(self, layer, params, embeds, scale=0.42):
        logger.info("Initializing to word2vec embeddings.")
        if not isinstance(params, list):
            params = [params]

        for pp in params:
            pv = pp.get_value()
            for i, v in embeds.items():
                pv[i] = scale*v
            layer.params[pp.name] = pv

    def __init_glove_embeds(self, layer, params, embeds):
        logger.info("Initializing to GLOVE embeddings.")
        if not isinstance(params, list):
            params = [params]

        glove_embs = self.emb_scale * embeds.astype("float32")
        mean = glove_embs.mean()
        std = glove_embs.std()

        token_embs = np.random.normal(loc=mean, scale=std, size=(2, 300))
        token_embs = np.concatenate([token_embs, glove_embs], axis=0)

        for pp in params:
            self.exclude_params[pp.name] = 1
            layer.params[pp.name] = token_embs.astype("float32")#, name=pp.name)

    def build_model(self,
                    use_noise=False,
                    mdl_name=None):

        if self.use_ff_controller:
            cls = NTMFFController
        else:
            cls = NTM

        if use_noise:
            mem_gater_activ = lambda x: self.mem_gater_activ(x, use_noise=use_noise)

        if self.use_bow_input and not self.bow_layer and not self.use_gru_inp_rep:
            self.bow_layer = BOWLayer(n_in=self.n_in,
                                      n_out=self.bow_size,
                                      seq_len=self.max_fact_len,
                                      weight_initializer=self.weight_initializer,
                                      bias_initializer=self.bias_initializer,
                                      use_average=False,
                                      name=self.pname("bow_layer"))
            if self.w2v_embed_path:
                fparams = self.bow_layer.params.lfilterby("weight")
                self.__init_to_embeds(self.bow_layer, fparams, self.w2v_embeds,
                                      scale=self.w2v_embed_scale)
        elif self.use_gru_inp_rep:
            if not self.gru_fact_layer_inps:
                low_cnames = ["low_reset_below",
                              "low_gater_below",
                              "low_state_below"]

                lnfout = len(low_cnames)
                self.low_cnames = map(lambda x: self.pname(x), low_cnames)
                self.gru_fact_layer_inps = ForkLayer(n_in=self.n_in,
                                                     n_outs=tuple([self.bow_size for i in xrange(lnfout)]),
                                                     weight_initializer=self.weight_initializer,
                                                     use_bias=False,
                                                     names=self.low_cnames)

                if self.w2v_embed_path:
                    fparams = self.gru_fact_layer_inps.params.lfilterby("weight")
                    self.__init_to_embeds(self.gru_fact_layer_inps, fparams, self.w2v_embeds)

            if not self.gru_fact_layer:
                self.gru_fact_layer = GRULayer(n_in=self.bow_size,
                                               n_out=self.bow_size,
                                               seq_len=self.max_fact_len,
                                               weight_initializer=self.weight_initializer,
                                               bias_initializer=self.bias_initializer,
                                               activ=Tanh,
                                               learn_init_state=self.learn_h0,
                                               name=self.pname("gru_fact_layer"))
        elif self.use_simple_rnn_inp_rep:

            if not self.rnn_fact_layer_inps:
                self.rnn_fact_layer_inps = AffineLayer(n_in=self.n_in,
                                                       n_out=self.bow_size,
                                                       weight_initializer=self.weight_initializer,
                                                       bias_initializer=self.bias_initializer,
                                                       name=self.pname("rnn_fact_layer_inps"))

                if self.w2v_embed_path:
                    fparams = self.rnn_fact_layer_inps.params.lfilterby("weight")
                    self.__init_to_embeds(self.rnn_fact_layer_inps, fparams, self.w2v_embeds)

            if not self.rnn_fact_layer:
                self.rnn_fact_layer = RNNLayer(n_in=self.n_in,
                                               n_out=self.bow_size,
                                               seq_len=self.max_fact_len,
                                               weight_initializer=self.weight_initializer,
                                               bias_initializer=self.bias_initializer,
                                               activ=Rect,
                                               learn_init_state=self.learn_h0,
                                               name=self.pname("rnn_fact_layer"))
        else:
            if not self.inp_proj_layer:
                self.inp_proj_layer = AffineLayer(n_in=self.n_in,
                                                n_out=self.bow_size,
                                                weight_initializer=self.weight_initializer,
                                                use_bias=False,
                                                bias_initializer=self.bias_initializer,
                                                name=self.pname("ntm_inp_proj_layer"))

                if self.glove_embed_path:
                    fparams = self.inp_proj_layer.params.lfilterby("weight")
                    self.__init_glove_embeds(self.inp_proj_layer,
                                             fparams,
                                             self.glove_embeds)

        if self.predict_bow_out and not self.bow_out_layer:
            self.bow_out_layer = AffineLayer(n_in=self.n_hids,
                                             n_out=self.n_out,
                                             weight_initializer=self.weight_initializer,
                                             noise=self.noise,
                                             wpenalty=self.wpenalty,
                                             bias_initializer=self.bias_initializer,
                                             name=self.pname("bow_out_layer"))

        if self.use_batch_norm and not self.batch_norm_layer:
            self.batch_norm_layer = BatchNormLayer(n_in=self.bow_size,
                                                   n_out=self.bow_size,
                                                   name=self.pname("batch_norm_inp"))

        if not self.ntm:
            inp = self.inps[0]
            bs = inp.shape[1]
            if inp.ndim == 4:
                bs = inp.shape[2]

            self.ntm = cls(n_in=self.bow_size,
                           n_hids=self.n_hids,
                           l1_pen=self.l1_pen,
                           learn_h0=self.learn_h0,
                           hybrid_att=self.hybrid_att,
                           smoothed_diff_weights=self.smoothed_diff_weights,
                           use_layer_norm=self.use_layer_norm,
                           recurrent_dropout_prob=self.recurrent_dropout_prob,
                           use_bow_input=self.use_bow_input,
                           use_loc_based_addressing=self.use_loc_based_addressing,
                           use_reinforce=self.use_reinforce,
                           erase_activ=self.erase_activ,
                           content_activ=self.content_activ,
                           mem_nel=self.mem_nel,
                           address_size=self.address_size,
                           use_context=self.use_context,
                           n_read_heads=self.n_read_heads,
                           use_soft_att=self.use_soft_att,
                           use_hard_att_eval=self.use_hard_att_eval,
                           use_inp_content=self.use_inp_content,
                           n_write_heads=self.n_write_heads,
                           dice_val=self.dice_val,
                           mem_size=self.mem_size,
                           use_nogru_mem2q=self.use_nogru_mem2q,
                           use_gru_inp_rep=self.use_gru_inp_rep,
                           weight_initializer=self.weight_initializer,
                           use_adv_indexing=self.use_adv_indexing,
                           wpenalty=self.wpenalty,
                           noise=self.noise,
                           n_layers=self.n_layers,
                           bias_initializer=self.bias_initializer,
                           use_quad_interactions=self.use_gate_quad_interactions,
                           controller_activ=self.controller_activ,
                           mem_gater_activ=self.mem_gater_activ,
                           batch_size=self.batch_size if self.batch_size else None,
                           use_multiscale_shifts=self.use_multiscale_shifts,
                           n_reading_steps=self.n_reading_steps,
                           seq_len=self.seq_len,
                           name=self.pname("ntm"),
                           use_noise=use_noise)

        if not self.merge_layer and self.use_deepout:
            self.merge_layer = MergeLayer(n_ins=[self.n_hids, self.mem_size],
                                          n_out=self.deep_out_size,
                                          weight_initializer=self.weight_initializer,
                                          bias_initializer=self.bias_initializer,
                                          names=[self.pname("deep_controller"),
                                                 self.pname("deep_mem")])

        if self.use_deepout:
            out_layer_in = self.deep_out_size
        else:
            out_layer_in = self.n_hids

        if self.use_out_mem:
            self.out_mem = AffineLayer(n_in=self.mem_size + self.address_size,
                                       n_out=self.n_out,
                                       weight_initializer=self.weight_initializer,
                                       wpenalty=self.wpenalty,
                                       noise=self.noise,
                                       bias_initializer=self.bias_initializer,
                                       name=self.pname("out_mem"))

            self.out_scaler = AffineLayer(n_in=self.n_hids,
                                          n_out=1,
                                          weight_initializer=self.weight_initializer,
                                          wpenalty=self.wpenalty,
                                          noise=self.noise,
                                          bias_initializer=self.bias_initializer,
                                          name=self.pname("out_scaler"))

        if not self.out_layer:
            self.out_layer = AffineLayer(n_in=out_layer_in,
                                         n_out=self.n_out,
                                         wpenalty=self.wpenalty,
                                         noise=self.noise,
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         name=self.pname("out"))

        if self.ntm.updates:
            self.updates.update(self.ntm.updates)


        if not self.use_reinforce_baseline and self.use_reinforce:
            self.baseline_out = AffineLayer(n_in=self.n_hids,
                                            n_out=1,
                                            weight_initializer=self.weight_initializer,
                                            bias_initializer=self.bias_initializer,
                                            init_bias_val=1e-3,
                                            name=self.pname("baseline_out"))

        if not self.children:
            self.children.append(self.ntm)
            if self.use_deepout and self.merge_layer:
                self.children.append(self.merge_layer)

            self.children.append(self.out_layer)
            if self.use_out_mem:
                self.children.extend([self.out_mem, self.out_scaler])

            if self.use_bow_input and self.bow_layer and not self.use_gru_inp_rep:
                self.children.append(self.bow_layer)
            elif self.use_gru_inp_rep:
                self.children.extend([self.gru_fact_layer_inps,
                                      self.gru_fact_layer])
            elif self.use_simple_rnn_inp_rep:
                self.children.extend([self.rnn_fact_layer_inps,
                                      self.rnn_fact_layer])
            else:
                self.children.append(self.inp_proj_layer)

            if self.predict_bow_out and self.bow_out_layer:
                self.children.append(self.bow_out_layer)

            if self.use_reinforce and not self.use_reinforce_baseline:
                self.children.append(self.baseline_out)

            if self.use_batch_norm:
                self.children.append(self.batch_norm_layer)

            self.merge_params()

            if self.renormalization_scale:
                self.params.renormalize_params(nscale=self.renormalization_scale,
                                               exclude_params=self.exclude_params)

        if mdl_name:
            logger.info("Reloading model from %s." % mdl_name)
            self.params.load(mdl_name)
            [child.use_params(self.params) for child in self.children]

        if self.trainpartitioner is None and self.sub_mb_size:
            self.trainpartitioner = MinibatchGradPartitioner(self.params,
                                                             self.sub_mb_size,
                                                             self.batch_size,
                                                             seq_len=self.seq_len)

    def get_cost(self,
                 use_noise=False,
                 valid_only=False,
                 mdl_name=None):

        probs, _ = self.fprop(use_noise=use_noise,
                              mdl_name=mdl_name)

        if isinstance(self.inps, list):
            X = self.inps[0]
            y = self.inps[1]
            if self.use_mask:
               mask = self.inps[2]
            cmask = None
            if self.use_cost_mask:
                cmask = self.inps[3]
        else:
            X = self.inps['x']
            y = self.inps['y']
            mask = self.inps['mask']
            cmask = None
            if self.use_cost_mask:
                cmask = self.inps['cmask']

        if self.l1_pen and self.l1_pen > 0 and not valid_only:
            self.reg += self.ntm.reg

        if self.l2_pen and not valid_only:
            self.l2_penalizer.penalize_layer_weights(self.out_layer)
            self.l2_penalizer.penalize_params(self.ntm.params.filterby("init_state").values[0])
            self.l2_penalizer.penalize_params(self.ntm.controller.params.filterby("weight").values[0])

            if not self.use_ff_controller:
                self.l2_penalizer.penalize_params(self.ntm.controller.params.filterby("state_before_ht").values[0])

            self.reg += self.l2_penalizer.get_penalty_level()

        if not self.softmax:
            self.cost = kl(y, probs, cost_mask=cmask)
            self.errors = 0
        else:
            if not self.use_last_hidden_state:
                self.cost, self.errors = nll(y, probs, cost_mask=cmask)
            else:
                self.cost, self.errors = nll(y, probs)

            if self.cost.ndim == 2:
                self.cost_mon = self.cost.sum(0).mean()

                if valid_only:
                    self.cost = self.cost_mon
            else:
                self.cost_mon = self.cost.mean()
                if valid_only:
                    self.cost = self.cost_mon

        bow_cost = 0
        if not valid_only:
            bow_cost_shifted = 0
            if self.predict_bow_out and self.bow_pred_out and self.bow_out_layer:
                bow_target = self.inps[-1]
                bcmask = mask * TT.cast(TT.eq(cmask, 0), "float32")
                sum_tru_time = False
                cost_matrix = True if self.use_reinforce and \
                        not sum_tru_time else False
                batch_vec = True if self.use_reinforce else False
                bow_cost = self.bow_out_w * kl(bow_target,
                                            self.bow_pred_out,
                                            batch_vec=batch_vec,
                                            sum_tru_time=sum_tru_time,
                                            cost_matrix=cost_matrix,
                                            cost_mask=bcmask,
                                            normalize_by_outsize=True)
                if cost_matrix:
                    bow_cost_shifted = TT.zeros_like(bow_cost)
                    bow_cost_shifted = TT.set_subtensor(bow_cost_shifted[1:], \
                            bow_cost[:-1])
                else:
                    bow_cost_shifted = bow_cost

            self.center = 0
            self.cost_std = 1

            if self.use_reinforce and self.use_reinforce_baseline:
                self.cost_mon = self.cost
                if not self.use_mask:
                    mask = None

                self.updates, self.known_grads, self.baseline, cost_std, \
                        self.write_policy, maxent_level = self.Reinforce(probs=self.write_weights,
                                                                        samples=self.w_samples,
                                                                        updates=self.updates,
                                                                        cost=(1 - self.bow_out_w) * self.cost + bow_cost_shifted,
                                                                        mask=mask)
                maxent_level = self.lambda2_reinf
            elif self.use_reinforce:
                if "float" in X.dtype:
                    self.baseline = self.baseline_out.fprop(self.ntm_outs[0]).reshape((X.shape[0],
                                                                                       X.shape[1])).dimshuffle(0,
                                                                                                               1,
                                                                                                               'x')
                else:
                    self.baseline = self.baseline_out.fprop(self.ntm_outs[0]).reshape((X.shape[1],
                                                                                       X.shape[2],
                                                                                       -1))

                mask_ = None
                mask = None
                if self.use_mask:
                    if mask:
                        mask_ = mask
                        if mask.ndim == 2:
                            mask_ = mask.dimshuffle(0, 1, 'x')
                        self.baseline = mask_ * self.baseline

                if not self.softmax:
                    self.cost = kl(y, probs, cost_mask=cmask, cost_matrix=True)
                    self.errors = 0
                else:
                    self.cost, self.errors = nll(y,
                                                 probs,
                                                 cost_mask=cmask,
                                                 cost_matrix=True)

                self.updates, self.known_grads, self.center, self.cost_std, \
                        self.write_policy, maxent_level = \
                            self.Reinforce(probs=self.write_weights,
                                           samples=self.w_samples,
                                           baseline=self.baseline,
                                           updates=self.updates,
                                           cost=(1 - self.bow_out_w) * self.cost + \
                                                bow_cost_shifted,
                                           mask=mask)
                if self.cost.ndim == 2:
                    hcost = self.cost.sum(0).dimshuffle('x', 0, 'x')
                else:
                    hcost = self.cost.dimshuffle(0, 'x', 'x')

                base_reg = huber_loss(y_hat=self.baseline,
                                      target=block_gradient(hcost),
                                      center=block_gradient(self.center),
                                      std=block_gradient(self.cost_std))

                if self.cost.ndim == 2:
                    self.cost_mon = self.cost.sum(0).mean()
                else:
                    self.cost_mon = self.cost.mean()

                if mask_:
                    base_reg = mask_ * base_reg

                self.base_reg = self.baseline_reg * base_reg.sum(0).mean()
                self.reg += self.base_reg

            if self.use_reinforce:
                self.ReaderReinforce.maxent_level = maxent_level
                self.read_constraint, self.read_policy = \
                                                self.ReaderReinforce(baseline=self.baseline,
                                                                     cost=self.cost + bow_cost,
                                                                     probs=self.read_weights,
                                                                     samples=self.r_samples,
                                                                     mask=mask,
                                                                     center=self.center,
                                                                     cost_std=self.cost_std)

            if self.cost.ndim == 2:
                self.cost = self.cost.sum(0).mean()
            else:
                self.cost = self.cost.mean()

            if bow_cost != 0 and bow_cost.ndim >= 1 and bow_cost != 0:
                bow_cost = bow_cost.sum(0).mean()

            if self.predict_bow_out and bow_cost:
                self.cost = (1 - self.bow_out_w) * self.cost + bow_cost

            if self.use_reinforce and self.read_constraint:
                self.cost += self.read_constraint

            if self.reg:
                self.cost += self.reg

        return self.cost, self.errors, bow_cost

    def get_inspect_fn(self,
                       mdl_name=None):

        logger.info("Compiling inspect function.")
        probs, ntm_outs = self.fprop(use_noise=False, mdl_name=mdl_name)
        updates = OrderedDict({})

        if self.ntm.updates and self.use_reinforce:
            updates.update(self.ntm.updates)

        inspect_fn = theano.function([self.inps[0], self.inps[2],
                                      self.inps[3], self.seq_len],
                                     ntm_outs + [probs],
                                     updates=self.ntm.updates,
                                     name=self.pname("inspect_fn"))
        return inspect_fn

    def get_valid_fn(self,
                     mdl_name=None):

        logger.info("Compiling validation function.")

        if self.predict_bow_out or self.bow_out_layer:
            if self.inps[-1].name == "bow_out":
                inps = self.inps[:-1]
        else:
            inps = self.inps

        if self.softmax:
            cost, errors, _ = self.get_cost(use_noise=True,
                                            valid_only=True,
                                            mdl_name=mdl_name)

            if self.ntm.updates:
                self.updates.update(self.ntm.updates)

            valid_fn = theano.function(inps + [self.seq_len],
                                       [cost, errors],
                                       updates=self.ntm.updates,
                                       on_unused_input='warn',
                                       name=self.pname("valid_fn"))

        else:
            cost, _, _ = self.get_cost(use_noise=False, mdl_name=mdl_name)
            if self.ntm.updates:
                self.updates.update(self.ntm.updates)

            valid_fn = theano.function(inps + [self.seq_len],
                                       [cost],
                                       updates=self.ntm.updates,
                                       on_unused_input='warn',
                                       name=self.pname("valid_fn"))

        return valid_fn

    def add_noise_to_params(self):
        for k, v in self.params.__dict__['params'].iteritems():
            v_np = v.get_value(borrow=True)
            noise = global_rng.normal(0, 0.05, v_np.shape)
            self.params[k] = v_np + noise

    def get_train_fn(self, lr=None, mdl_name=None):
        if lr is None:
            lr = self.eps

        if self.softmax:
            cost, errors, bow_cost = self.get_cost(use_noise=True,
                                                   mdl_name=mdl_name)
        else:
            cost, _, _ = self.get_cost(use_noise=True,
                                       mdl_name=mdl_name)

        params = self.params.values
        logger.info("Computing the gradients.")
        self.grads_timer.start()

        inps = self.inps
        if self.predict_bow_out:
            inps = self.inps + [self.bow_out_w]
        if not self.learn_embeds:
            params.pop(0)

        grads = safe_grad(cost, params, known_grads=self.known_grads)
        self.grads_timer.stop()
        logger.info(self.grads_timer)

        logger.info("Compiling grad fn.")
        self.gradfn_timer.start()

        if self.sub_mb_size:
            if self.sub_mb_size != self.batch_size:
                self.comp_grad_fn, grads = self.trainpartitioner.get_compute_grad_fn(grads,
                                                                                     self.ntm.updates,
                                                                                     inps)

        gnorm = sum(grad.norm(2) for _, grad in grads.iteritems())
        updates, norm_up, param_norm = self.learning_rule.get_updates(learning_rate=lr,
                                                                      grads=grads)

        self.gradfn_timer.stop()
        logger.info(self.gradfn_timer)

        if self.updates:
            self.updates.update(updates)
        else:
            self.updates = updates
            warnings.warn("WARNING: Updates are empty.")

        logger.info("Compiling the training function.")
        self.train_timer.start()
        if hasattr(self, "cost_mon"):
            outs = [self.cost_mon, gnorm, norm_up, param_norm]
        else:
            outs = [cost, gnorm, norm_up, param_norm]

        if self.softmax:
            outs += [self.errors]

        if self.predict_bow_out:
            outs += [bow_cost]

        if self.use_reinforce:
            outs += [self.read_constraint, self.baseline, self.read_policy, \
                                                           self.write_policy]
            if not self.use_reinforce_baseline:
                outs += [self.center, self.cost_std, self.base_reg]

        if self.use_batch_norm:
            self.updates.update(self.batch_norm_layer.updates)

        train_fn = theano.function(inps + [self.seq_len],
                                   outs,
                                   updates=self.updates,
                                   mode=self.theano_function_mode,
                                   name=self.pname("train_fn"))

        self.train_timer.stop()
        logger.info(self.train_timer)

        if self.train_profile:
            import sys
            sys.exit(-1)

        return train_fn

    def fprop(self, inps=None,
              leak_rate=0.05,
              use_noise=False,
              mdl_name=None):

        self.build_model(use_noise=use_noise, mdl_name=mdl_name)
        self.ntm.evaluation_mode = use_noise
        if not inps:
            inps = self.inps

        # First two are X and targets
        # assert (2 + sum([use_mask, use_cmask])) + 1 >= len(inps), \
        #    "inputs have illegal shape."
        cmask = None
        mask = None
        if isinstance(inps, list):
            X = inps[0]
            y = inps[1]

            if self.use_mask:
                mask = inps[2]
                if self.use_cost_mask:
                    cmask = inps[3]
        else:
            X = inps['X']
            y = inps['y']
            if self.use_mask:
                mask = inps['mask']
                if self.use_cost_mask:
                    cmask = inps['cmask']

        if self.use_cost_mask:
            if cmask is not None:
                if self.use_bow_cost_mask:
                    if mask.ndim == cmask.ndim:
                        m = (mask * TT.eq(cmask, 0)).reshape((cmask.shape[0] * cmask.shape[1], -1))
                    else:
                        m = (mask.dimshuffle(0, 1, 'x') * TT.eq(cmask, 0))[:, :, 0].reshape((cmask.shape[0] * cmask.shape[1], -1))
                else:
                    m = mask
            else:
                raise ValueError("Mask for the answers should not be empty.")

        if X.ndim == 2 and y.ndim == 1:
            # For sequential MNIST.
            if self.permute_order:
                X = X.dimshuffle(1, 0)
                idxs = self.rnd_indxs
                X = X[idxs]
            inp_shp = (X.shape[0], X.shape[1], -1)
        else:
            inp_shp = (X.shape[1], X.shape[2], -1)

        self.ntm_in = None
        if self.use_bow_input and not self.use_gru_inp_rep and not self.use_simple_rnn_inp_rep:

            bow_out = self.bow_layer.fprop(X, amask=m, deterministic=not use_noise)
            bow_out = bow_out.reshape((X.shape[1], X.shape[2], -1))
            self.ntm_in = bow_out

        elif self.use_gru_inp_rep:
            m0 = as_floatX(TT.gt(X, 0))
            if self.use_mask and self.use_cost_mask:
                if cmask is not None:
                    m1 = mask * TT.eq(cmask, 0)
                else:
                    raise ValueError("Mask for the answers should not be empty.")

            low_inp_shp = (X.shape[0], X.shape[1] * X.shape[2], -1)
            Xr = X.reshape(low_inp_shp)
            grufact_inps = self.gru_fact_layer_inps.fprop(Xr)
            low_reset_below = grufact_inps.values()[0].reshape(low_inp_shp)
            low_gater_below = grufact_inps.values()[1].reshape(low_inp_shp)
            low_state_below = grufact_inps.values()[2].reshape(low_inp_shp)
            linps = [low_reset_below, low_gater_below, low_state_below]

            m0_part = TT.cast(m0.sum(0).reshape((X.shape[1],
                                                 X.shape[2])).dimshuffle(0, 1, 'x'), 'float32')
            m0_part = TT.switch(TT.eq(m0_part, as_floatX(0)), as_floatX(1), m0_part)

            h0 = self.gru_fact_layer.fprop(inps=linps,
                                           mask=m0,
                                           batch_size=self.batch_size)

            self.ntm_in = m1.dimshuffle(0, 1, 'x') * ((m0.dimshuffle(0, 1, 2, 'x') * h0.reshape((X.shape[0],
                                                                   X.shape[1],
                                                                   X.shape[2],
                                                                   -1))).sum(0) \
                                                                           / m0_part).reshape(inp_shp)
        elif self.use_simple_rnn_inp_rep:
            m0 = as_floatX(TT.gt(X, 0))
            if cmask is not None:
                m1 = mask * TT.eq(cmask, 0)
            else:
                raise ValueError("Mask for the answers should not be empty.")

            low_inp_shp = (X.shape[0], X.shape[1] * X.shape[2], -1)
            Xr = X.reshape(low_inp_shp)
            rnnfact_inps = self.rnn_fact_layer_inps.fprop(Xr).reshape(low_inp_shp)
            m0 = m0.reshape(low_inp_shp)

            h0 = self.rnn_fact_layer.fprop(inps=rnnfact_inps,
                                           mask=m0,
                                           batch_size=self.batch_size)

            m0_part = TT.cast(m0.sum(0).reshape((X.shape[1],
                                                X.shape[2])).dimshuffle(0,
                                                                        1,
                                                                       'x'), 'float32')

            m0_part = TT.switch(m0_part == 0, as_floatX(1), m0_part)
            self.ntm_in = m1.dimshuffle(0, 1, 'x') * (h0.reshape((X.shape[0],
                                                                  X.shape[1],
                                                                  X.shape[2],
                                                                  -1)).sum(0) / \
                                                                          m0_part).reshape(inp_shp)

        else:
            X_proj = self.inp_proj_layer.fprop(X)
            if not self.learn_embeds:
                X_proj = block_gradient(X_proj)

            if self.use_batch_norm:
                X_proj = self.batch_norm_layer.fprop(X_proj,
                                                     inference=not use_noise)
            self.ntm_in = X_proj

        context = None
        if self.use_context:
            if self.use_qmask:
                context = (self.qmask.dimshuffle(0, 1, 'x') * self.ntm_in).sum(0)
            else:
                m1_part = m1.sum(0).dimshuffle(0, 'x')
                context = self.ntm_in.sum(0) / m1_part

        self.ntm_outs = self.ntm.fprop(self.ntm_in,
                                       mask=mask,
                                       cmask=cmask,
                                       context=context,
                                       batch_size=self.batch_size,
                                       use_mask=self.use_mask,
                                       use_noise=not use_noise)

        h, m_read = self.ntm_outs[0], self.ntm_outs[2]

        if self.use_reinforce:
            self.w_samples, self.r_samples = self.ntm_outs[-2], self.ntm_outs[-1]

            if self.smoothed_diff_weights:
                idx = -6
            else:
                idx = -4

            self.write_weights, self.read_weights = self.ntm_outs[idx], \
                    self.ntm_outs[idx+1]
        else:
            self.write_weights, self.read_weights = self.ntm_outs[3], self.ntm_outs[4]

        if self.anticorrelation:
            acorr = AntiCorrelationConstraint(level=self.anticorrelation)
            rw1 = self.read_weights[:, 0]
            rw2 = self.read_weights[:, 1]
            self.reg += acorr(rw1, rw2, mask=mask)

        if self.correlation_ws:
            logger.info("Applying the correlation constraint.")
            corr_cons = CorrelationConstraint(level=self.correlation_ws)
            self.reg += corr_cons(self.read_weights, self.write_weights, mask,
                                  self.qmask)

        if self.use_last_hidden_state:
            h = h.reshape(inp_shp)
            h = h[-1]

        if self.use_deepout:
            merged_out = self.merge_layer.fprop([h, m_read])
            out_layer = Leaky_Rect(merged_out, leak_rate)

            if self.dropout:
                dropOp = Dropout(dropout_prob=self.dropout)
                out_layer = dropOp(out_layer, deterministic=not use_noise)

            out_layer = self.out_layer.fprop(out_layer,
                                             deterministic=not use_noise)
        else:
            if self.use_out_mem:
                if self.dropout:
                    dropOp = Dropout(dropout_prob=self.dropout)
                    m_read = dropOp(m_read, deterministic=not use_noise)

                mem_out = self.out_mem.fprop(m_read, deterministic=not use_noise)
                mem_scaler = self.out_scaler.fprop(h,
                                                   deterministic=not use_noise).reshape((
                                                   mem_out.shape[0],)).dimshuffle(0, 'x')

                h_out = self.out_layer.fprop(h, deterministic=not use_noise)
                out_layer = h_out + mem_out * Sigmoid(mem_scaler)
            else:
                if self.dropout:
                    dropOp = Dropout(dropout_prob=self.dropout)
                    h = dropOp(h, deterministic=not use_noise)
                out_layer = self.out_layer.fprop(h,
                                                 deterministic=not use_noise)

        if self.predict_bow_out and self.bow_out_layer:
            logger.info("Using the bow output prediction.")
            self.bow_pred_out = Sigmoid(self.bow_out_layer.fprop(h,
                                        deterministic=not use_noise))

        if self.softmax:
            self.probs = Softmax(out_layer)
        else:
            self.probs = Sigmoid(out_layer)

        if self.ntm.updates:
            self.updates.update(self.ntm.updates)

        self.str_params(logger)

        self.h = h
        return self.probs, self.ntm_outs

    def __get_state__(self):
        return self.state

    def __set_state__(self, state):
        self.__dict__.update(state)

