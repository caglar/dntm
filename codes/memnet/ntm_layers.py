import logging

import numpy as np
import theano
import theano.tensor as TT

from collections import OrderedDict

from core.parameters import Parameters, BiasInitializer, BiasInitMethods
from core.commons import (Rect, Trect,
                                 Tanh, Sigmoid,
                                 EPS, global_trng)

from core.commons import floatX
from core.utils import safe_izip, concatenate, sample_weights_classic, \
                                const, as_floatX, overrides, block_gradient
from core.utils.nnet_utils import get_hard_vals

from core.layers import (Layer, RecurrentLayer,
                                AffineLayer, ForkLayer,
                                QuadraticInteractionLayer,
                                PowerupLayer, MergeLayer)

from core.operators import MemorySimilarity, CircularConvolve, \
        CircularConvolveAdvIndexing, GeomEuclideanSigmoidDot

from memory import AddressedMemory
from controllers import *
from addresser import Addresser


logger = logging.getLogger(__name__)
logger.disabled = False


class NTMBase(RecurrentLayer):

    def __init__(self, batch_size, learn_h0, evaluation_mode):
        self.batch_size = batch_size
        self.learn_h0 = learn_h0
        self.evaluation_mode = evaluation_mode
        super(NTMBase, self).__init__()

    def _new_state(self, shp, name=None, adapt_state=True):
        if self.learn_h0 and adapt_state:
            assert self.params is not None
            if shp[0] == self.batch_size:
                self.params[name] = (np.zeros(shp[1:]) + 1e-6).astype("float32")
                new_state = TT.alloc(self.params[name], *shp)
                new_state.name = name
            else:
                if len(shp) == 1:
                    self.params[name] = (np.zeros(shp) + 1e-6).astype("float32")
                    new_state = self.params[name]
                    new_state.name = name
                elif len(shp) == 2:
                    self.params[name] = (np.zeros(shp[1:]) + 1e-6).astype("float32")
                    new_state = TT.alloc(self.params[name], *shp)
                    new_state.name = name
                elif len(shp) == 3:
                    self.params[name] = (np.zeros((shp[0], shp[2])) + 1e-6).astype("float32")
                    new_state = TT.alloc(self.params[name].dimshuffle(0, 'x', 1), *shp)
                    new_state.name = name
        else:
            new_state = TT.alloc(as_floatX(0), *shp)
            new_state.name = name
        return new_state

    def _new_stateff(self, shp, name=None, adapt_state=True):
        if self.learn_h0 and adapt_state:
            if shp[0] == self.batch_size:
                self.params[name] = np.zeros(shp[1:])
                new_state = TT.concatenate([[self.params[name]] for i in xrange(self.batch_size)],
                                            axis=0).reshape(shp)
            else:
                self.params[name] = np.zeros(shp)
                new_state = self.params[name]
        else:
            new_state = TT.alloc(as_floatX(0.), *shp)
            new_state.name = name
        return new_state

    def _get_sample_weights(self, weights):
        if weights.ndim == 3:
            weights = weights.reshape((weights.shape[0], weights.shape[1]))
        if self.evaluation_mode:
            weight_idxs = get_hard_vals(weights, tot_size=self.mem_nel)
        else:
            if self.hybrid_att:
                rnd_val = global_trng.multinomial(pvals=weights, dtype=floatX)
                weight_idxs = self.dice * weights  + (1. - self.dice) * rnd_val
            else:
                weight_idxs = global_trng.multinomial(pvals=weights, dtype=floatX)

        if self.use_soft_att:
            print "Use soft attention!!!!"
            weight_idxs = weights

        return weight_idxs


class NTMFFController(NTMBase):
    """
        A simple Neural Turing Machine implementation.
    """
    def __init__(self,
                 n_in=None,
                 n_hids=None,
                 use_multiscale_shifts=True,
                 mem_nel=None,
                 smoothed_diff_weights=False,
                 learn_h0=False,
                 mem_size=None,
                 weight_initializer=None,
                 use_loc_based_addressing=True,
                 use_reinforce=False,
                 bias_initializer=None,
                 controller_activ=None,
                 use_lstm_controller=False,
                 use_bow_input=False,
                 recurrent_dropout_prob=-1,
                 use_layer_norm=False,
                 use_gru_inp_rep=False,
                 n_reading_steps=1,
                 multi_step_q_only=True,
                 use_inp_content=True,
                 n_layers=2,
                 hybrid_att=True,
                 use_context=False,
                 mem_gater_activ=None,
                 address_size=None,
                 use_quad_interactions=False,
                 use_nogru_mem2q=False,
                 erase_activ=None,
                 use_adv_indexing=True,
                 wpenalty=None,
                 noise=None,
                 use_soft_att=False,
                 use_hard_att_eval=False,
                 evaluation_mode=False,
                 l1_pen=None,
                 n_read_heads=1,
                 n_write_heads=1,
                 sampling=False,
                 dice_val=None,
                 batch_size=32,
                 content_activ=None,
                 seq_len=None,
                 use_noise=False,
                 name="gru_nmt"):

        if n_in is None:
            raise ValueError("Number of inputs should not be empty.")

        if n_read_heads is None or n_write_heads <= 0:
            raise ValueError("Number of heads should be positive and it should not be None.")

        print "Address_size is", address_size
        self.wpenalty = wpenalty
        self.noise = noise

        # For now this part of the code has no functionality:
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.use_layer_norm = use_layer_norm

        self.use_gru_inp_rep = use_gru_inp_rep
        self.use_multiscale_shifts = use_multiscale_shifts
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.use_reinforce = use_reinforce
        self.n_reading_steps = n_reading_steps
        self.multi_step_q_only = multi_step_q_only
        self.smoothed_diff_weights = smoothed_diff_weights
        self.use_context = use_context
        self.n_layers = n_layers
        self.hybrid_att = hybrid_att
        self.use_soft_att = use_soft_att
        self.use_hard_att_eval = use_hard_att_eval

        if hybrid_att:
            print "Using hybrid att!!"
            if dice_val:
                self.dice = global_trng.binomial((1,), p=dice_val,
                                                 n=1, dtype="float32").sum()
            else:
                self.dice = global_trng.binomial((1,), p=0.5,
                                                 n=1, dtype="float32").sum()

        # TODO: Support quadratic interactions for this
        self.use_quad_interactions = False

        if controller_activ is not None and isinstance(controller_activ, str):
            self.controller_activ = eval(controller_activ)
        elif controller_activ:
            self.controller_activ = controller_activ
        else:
            self.controller_activ = Tanh

        if mem_gater_activ is not None and isinstance(mem_gater_activ, str):
            self.mem_gater_activ = eval(mem_gater_activ)
        elif mem_gater_activ is not None:
            self.mem_gater_activ = mem_gater_activ
        else:
            self.mem_gater_activ = Sigmoid

        self.use_loc_based_addressing = use_loc_based_addressing
        self.updates = OrderedDict()

        super(NTMFFController, self).__init__(learn_h0=learn_h0,
                                              batch_size=batch_size,
                                              evaluation_mode=evaluation_mode)
        self.n_in = n_in
        self.n_hids = n_hids
        self.mem_nel = mem_nel
        self.mem_size = mem_size
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.use_noise = use_noise
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.address_size = address_size if address_size else 0
        self.l1_pen = l1_pen
        self.use_inp_content = use_inp_content
        self.sampling = sampling
        self.use_adv_indexing = use_adv_indexing
        self.use_bow_input = use_bow_input
        self.use_lstm_controller = use_lstm_controller
        self.time_idxs = theano.shared(as_floatX(np.arange(self.mem_nel)), name="time_idxs")

        logger.info("Size of the address is %d" % self.address_size)
        logger.info("Learning the first state %d " % self.learn_h0)

        self.name = name
        self.outputs_info = []

        self.content_activ = content_activ or Trect
        self.erase_activ = erase_activ or Trect

        self.seq_len = seq_len
        self.init_params()
        self.__create_states()

    def __create_states(self, inp=None):

        if inp:
            bs = inp.shape[1]
            if inp.ndim == 4:
                bs = inp.shape[2]
        else:
            bs = self.batch_size

        if not self.sampling and self.batch_size > 1:
            mem_state = TT.alloc(self.memory.M.dimshuffle('x', 0, 1),
                                 bs,
                                 self.memory.M.shape[0],
                                 self.memory.M.shape[1])

            mem_read = self._new_state((bs,
                                        self.mem_size + self.address_size),
                                        name=self.pname("mem_read"),
                                        adapt_state=False)

            if self.n_write_heads > 1:
                write_weights = self._new_state((self.n_write_heads,
                                                 bs,
                                                 self.mem_nel),
                                                 name=self.pname("write_weights"),
                                                 adapt_state=False)
            else:
                write_weights = self._new_state((bs, self.mem_nel),
                                                 name=self.pname("write_weights"),
                                                 adapt_state=False)

            self.outputs_info.extend(
                [None, mem_state, mem_read, write_weights])

            if self.n_read_heads > 1:
                read_weights = self._new_state((self.n_read_heads,
                                                bs,
                                                self.mem_nel),
                                                name=self.pname("read_weights"),
                                                adapt_state=False)
            else:
                if self.n_reading_steps > 1:
                    read_weights = self._new_state((self.n_reading_steps,
                                                    bs,
                                                    self.mem_nel),
                                                    name=self.pname("read_weights"),
                                                    adapt_state=False)
                else:
                    read_weights = self._new_state((bs,
                                                    self.mem_nel),
                                                    name=self.pname("read_weights"),
                                                    adapt_state=False)

            self.outputs_info.append(read_weights)
        else:
            mem_state = self.memory.M
            mem_read = self._new_state((self.mem_size + self.address_size),
                                        name=self.pname("mem_read"),
                                        adapt_state=False)

            if self.n_write_heads > 1:
                write_weights = self._new_state((self.n_write_heads, self.mem_nel),
                                                 name=self.pname("write_weights"),
                                                 adapt_state=False)
            else:
                write_weights = self._new_state((self.mem_nel),
                                                 name=self.pname("write_weights"),
                                                 adapt_state=False)
            self.outputs_info.extend([None, mem_state, mem_read, write_weights])

            if self.n_read_heads > 1:
                read_weights = self._new_state((self.n_read_heads, self.mem_nel),
                                                name=self.pname("read_weights"),
                                                adapt_state=False)
            else:
                read_weights = self._new_state((self.mem_nel,),
                                                name=self.pname("read_weights"),
                                                adapt_state=False)
            self.outputs_info.append(read_weights)

        if self.use_reinforce:
            self.outputs_info.extend([None, None])

        if self.smoothed_diff_weights:
            if self.n_read_heads > 1:
                self.read_weights_pre = self._new_state((self.n_read_heads,
                                                         bs,
                                                         self.mem_nel),
                                                         name=self.pname("read_weights_pre"),
                                                         adapt_state=True)
            else:
                if self.n_reading_steps > 1:
                    self.read_weights_pre = self._new_state((self.n_reading_steps,
                                                    bs,
                                                    self.mem_nel),
                                                    name=self.pname("read_weights_pre"),
                                                    adapt_state=True)
                else:
                    self.read_weights_pre = self._new_state((bs,
                                                    self.mem_nel),
                                                    name=self.pname("read_weights_pre"),
                                                    adapt_state=True)

            if self.n_write_heads > 1:
                self.write_weights_pre = self._new_state((self.n_write_heads,
                                                          bs,
                                                          self.mem_nel),
                                                          name=self.pname("write_weights_pre"),
                                                          adapt_state=True)
            else:
                self.write_weights_pre = self._new_state((bs, self.mem_nel),
                                                         name=self.pname("write_weights_pre"),
                                                         adapt_state=True)
            self.outputs_info.extend([self.write_weights_pre,
                                      self.read_weights_pre])


    def init_params(self):
        self.ff_controller = FFController(n_hids=self.n_hids,
                                          mem_size=self.mem_size + self.address_size,
                                          weight_initializer=self.weight_initializer,
                                          bias_initializer=self.bias_initializer,
                                          n_layers=self.n_layers,
                                          activ=self.controller_activ,
                                          name=self.pname("ff_controller"))

        self.erase_heads = [ AffineLayer(n_in=self.n_hids,
                                         n_out=self.mem_size,
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         name=self.pname("erase_head_ht_0")) ]

        self.content_heads = [ AffineLayer(n_in=self.n_hids,
                                           n_out=self.mem_size,
                                           weight_initializer=self.weight_initializer,
                                           bias_initializer=self.bias_initializer,
                                           name=self.pname("content_head_0")) ]

        self.write_heads = [ Addresser(n_hids=self.n_hids,
                                       mem_size=self.mem_size,
                                       mem_nel=self.mem_nel,
                                       address_size=self.address_size,
                                       smoothed_diff_weights=self.smoothed_diff_weights,
                                       use_reinforce=self.use_reinforce,
                                       use_loc_based_addressing=self.use_loc_based_addressing,
                                       use_multiscale_shifts=self.use_multiscale_shifts,
                                       use_adv_indexing=self.use_adv_indexing,
                                       n_mid_key_size=self.n_hids,
                                       weight_initializer=self.weight_initializer,
                                       bias_initializer=self.bias_initializer,
                                       name=self.pname("addresser_head_0")) ]

        for i in xrange(1, self.n_write_heads):
            self.write_heads.append(Addresser(n_hids=self.n_hids,
                                             mem_size=self.mem_size,
                                             mem_nel=self.mem_nel,
                                             address_size=self.address_size,
                                             smoothed_diff_weights=self.smoothed_diff_weights,
                                             use_loc_based_addressing=self.use_loc_based_addressing,
                                             use_adv_indexing=self.use_adv_indexing,
                                             use_reinforce=self.use_reinforce,
                                             use_multiscale_shifts=self.use_multiscale_shifts,
                                             n_mid_key_size=self.n_hids,
                                             weight_initializer=self.weight_initializer,
                                             bias_initializer=self.bias_initializer,
                                             name=self.pname("addresser_head_%d" % i)))

            self.erase_heads.append(AffineLayer(n_in=self.n_hids,
                                                n_out=self.mem_size,
                                                weight_initializer=self.weight_initializer,
                                                bias_initializer=self.bias_initializer,
                                                name=self.pname("erase_head_ht_%d" % i)))

            self.content_heads.append(AffineLayer(n_in=self.n_hids,
                                                  n_out=self.mem_size,
                                                  weight_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  name=self.pname("content_head_%d" % i)))


        self.controller_inp = AffineLayer(n_in=self.n_in,
                                          n_out=self.n_hids,
                                          weight_initializer=self.weight_initializer,
                                          bias_initializer=self.bias_initializer,
                                          use_bias=False,
                                          name=self.pname("controller_inp"))


        self.children = [self.ff_controller, self.controller_inp] + \
                         self.write_heads +  self.erase_heads + self.content_heads

        if self.use_inp_content:
            self.inp2_content = AffineLayer(n_in=self.n_hids,
                                            n_out=self.mem_size,
                                            weight_initializer=self.weight_initializer,
                                            bias_initializer=self.bias_initializer,
                                            name=self.pname("inp2_content"))

            self.inp_scale_content = AffineLayer(n_in=self.n_hids,
                                                 n_out=1,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 name=self.pname("inp_scale_2content"))

            self.children.extend([self.inp2_content, self.inp_scale_content])

        if self.use_context:
            self.context_proj = AffineLayer(n_in=self.n_in,
                                            n_out=self.n_hids,
                                            weight_initializer=self.weight_initializer,
                                            bias_initializer=self.bias_initializer,
                                            name=self.pname("context_proj"))
            self.children += [self.context_proj]

        self.read_heads = [Addresser(n_hids=self.n_hids,
                                     use_loc_based_addressing=self.use_loc_based_addressing,
                                     mem_size=self.mem_size,
                                     mem_nel=self.mem_nel,
                                     address_size=self.address_size,
                                     n_mid_key_size=self.n_hids,
                                     smoothed_diff_weights=self.smoothed_diff_weights,
                                     use_adv_indexing=self.use_adv_indexing,
                                     use_reinforce=self.use_reinforce,
                                     use_multiscale_shifts=self.use_multiscale_shifts,
                                     weight_initializer=self.weight_initializer,
                                     bias_initializer=self.bias_initializer,
                                     name=self.pname("read_addresser_head_0"))]

        for i in xrange(1, self.n_read_heads):
            self.read_heads.append(Addresser(n_hids=self.n_hids,
                                             mem_size=self.mem_size,
                                             mem_nel=self.mem_nel,
                                             smoothed_diff_weights=self.smoothed_diff_weights,
                                             address_size=self.address_size,
                                             use_loc_based_addressing=self.use_loc_based_addressing,
                                             n_mid_key_size=self.n_hids,
                                             use_adv_indexing=self.use_adv_indexing,
                                             use_reinforce=self.use_reinforce,
                                             use_multiscale_shifts=self.use_multiscale_shifts,
                                             weight_initializer=self.weight_initializer,
                                             bias_initializer=self.bias_initializer,
                                             name=self.pname("read_addresser_head_%d" % i)))

        self.children += self.read_heads

        self.memory = AddressedMemory(mem_nel=self.mem_nel,
                                      mem_size=self.mem_size,
                                      address_size=self.address_size,
                                      n_read_heads=self.n_read_heads,
                                      n_write_heads=self.n_write_heads,
                                      n_reading_steps=self.n_reading_steps)

        self.children += [self.memory]
        self.merge_params()

    def __get_writer_weights(self,
                             h_t=None,
                             state_below=None,
                             write_weight_before=None,
                             write_weight_before_pre=None,
                             mem_before=None,
                             time_idxs=None):

        first_write_weight = write_weight_before[0] if self.n_write_heads > 1 \
                else write_weight_before
        first_write_weight_pre = write_weight_before_pre[0] if self.n_write_heads > 1 \
                else write_weight_before_pre
        write_heads_w, write_weights_pre = \
                self.write_heads[0].fprop(h_t,
                                          mem_before[:, 1:, :],
                                          first_write_weight,
                                          w_t_pre_before=first_write_weight_pre,
                                          time_idxs=time_idxs)

        for i in xrange(1, self.n_write_heads):
            write_weight_tmp, write_weights_pre_tmp = \
                    self.write_heads[i].fprop(h_t,
                                              mem_before[:, 1:, :],
                                              write_weight_before[i],
                                              w_t_pre_before=write_weight_before_pre[i],
                                              time_idxs=time_idxs)
            write_heads_w.extend(write_weight_tmp)
            write_weights_pre.extend(write_weights_pre)

        if not self.use_reinforce and self.evaluation_mode and self.use_hard_att_eval:
            print("Using discrete attention during the evaluation...")
            if isinstance(write_heads_w, list):
                write_heads_w = [ get_hard_vals(ww, tot_size=self.mem_nel) for ww in write_heads_w ]
            else:
                write_heads_w = get_hard_vals(write_heads_w, tot_size=self.mem_nel)

        # TODO: change mem_read_before to mem_before
        erase_heads = [self.erase_activ(erase.fprop(h_t))
                            for erase in self.erase_heads]

        if self.use_inp_content:
            inp2_content = self.inp2_content.fprop(state_below)
            inp_scaler = self.inp_scale_content.fprop(h_t).reshape((inp2_content.shape[0],)).dimshuffle(0, 'x')
            inp_proj = inp2_content * Sigmoid(inp_scaler)
            contents = [self.content_activ(content.fprop(h_t) + inp_proj) for content in self.content_heads]
        else:
            contents = [self.content_activ(content.fprop(h_t)) for content in self.content_heads]

        if erase_heads[0].ndim == 2:
            erase_heads = [erase_head.dimshuffle(0, 'x', 1) for erase_head in erase_heads]
            write_heads_w = [w.dimshuffle(0, 1, 'x') for w in write_heads_w]
            contents = [content.dimshuffle(0, 'x', 1) for content in contents]
            if self.smoothed_diff_weights:
                write_weights_pre = TT.concatenate([wwp.dimshuffle('x', 0, 1) for wwp \
                        in write_weights_pre])
            if mem_before.ndim == 2:
                M_ = mem_before.dimshuffle('x', 0, 1)
            else:
                M_ = mem_before
        else:
            raise ValueError("erase_ht should be 2D!!!!")

        #Writing Stage:
        write_weights_samples = write_heads_w
        if self.use_reinforce:
            write_weights_samples = [self._get_sample_weights(w) for w in write_heads_w]

        write_weights, write_weights_samples, m_t = self.memory.write(erase_heads, \
                write_weights_samples, write_heads_w, contents, M_)
        return write_weights, write_weights_samples, m_t, write_heads_w, write_weights_pre

    def __get_reader_weights_single_step(self,
                                         h_t=None,
                                         mem_before=None,
                                         write_heads_w=None,
                                         read_weight_before=None,
                                         read_samples_before=None,
                                         read_weight_before_pre=None,
                                         time_idxs=None):

            m_t = mem_before
            if self.n_read_heads > 1:
                first_read_weight = read_weight_before[0]
                first_read_weight_pre = read_weight_before_pre[0] if self.smoothed_diff_weights \
                        else None
            elif self.n_reading_steps > 1:
                first_read_weight = read_weight_before[-1]
                first_read_weight_pre = read_weight_before_pre[-1] if self.smoothed_diff_weights \
                        else None
            else:
                first_read_weight = read_weight_before
                first_read_weight_pre = read_weight_before_pre if self.smoothed_diff_weights \
                        else None

            read_heads_w, read_pre_weights = \
                    self.read_heads[0].fprop(h_t,
                                             m_t[:, :-1, :],
                                             first_read_weight,
                                             w_t_pre_before=first_read_weight_pre,
                                             time_idxs=time_idxs)

            for i in xrange(1, self.n_read_heads):
                read_heads_w_tmp, read_pre_weights_tmp = \
                        self.read_heads[i].fprop(h_t,
                                                 m_t[:, :-1, :],
                                                 read_weight_before[i],
                                                 w_t_pre_before=read_weight_before_pre[i],
                                                 time_idxs=time_idxs)

                read_heads_w.extend(read_heads_w_tmp)
                read_pre_weights.extend(read_pre_weights_tmp)

            if not self.use_reinforce and self.evaluation_mode and self.use_hard_att_eval:
                print("Using discrete attention during the evaluation...")
                if isinstance(read_heads_w, list):
                    read_heads_w = [ get_hard_vals(rw, tot_size=self.mem_nel) for rw in read_heads_w ]
                else:
                    read_heads_w = get_hard_vals(read_heads_w, tot_size=self.mem_nel)


            read_weights_samples = read_heads_w

            if self.use_reinforce:
                read_weights_samples = [self._get_sample_weights(w) \
                                        for w in read_weights_samples]

            read_weights, read_weights_samples, mem_read_t = \
                    self.memory.read(read_heads_w, read_weights_samples, m_t)

            return read_weights, read_weights_samples, mem_read_t, read_pre_weights

    def __get_reader_weights(self,
                             h_t=None,
                             state_below=None,
                             mem_before=None,
                             write_heads_w=None,
                             read_weight_before=None,
                             read_weight_before_pre=None,
                             time_idxs=None):

        read_weights_steps = []
        read_samples_steps = []
        read_ws_pre_steps = []
        add_to_list = lambda x, y:  x.extend(y) if isinstance(y, list) else x.append(y)

        for i in xrange(self.n_reading_steps):
            read_weights, read_samples, mem_read, read_pre_weights = \
                self.__get_reader_weights_single_step(h_t,
                                                      mem_before=mem_before,
                                                      write_heads_w=write_heads_w,
                                                      read_weight_before=read_weight_before,
                                                      read_weight_before_pre=read_weight_before_pre,
                                                      time_idxs=time_idxs)

            add_to_list(read_weights_steps, read_weights)
            add_to_list(read_samples_steps, read_samples)
            add_to_list(read_ws_pre_steps, read_pre_weights)

            if i < self.n_reading_steps - 1:
                read_weight_before = read_weights
                mem_read_before = mem_read

                h_t = self.ff_controller.fprop(state_below=state_below,
                                               mem_before=mem_read_before)

        if self.n_reading_steps is not None and self.n_reading_steps > 1:
            read_weights = TT.concatenate([r.dimshuffle('x', 0, 1) for r in \
                    read_weights_steps])
            read_samples = TT.concatenate([r.dimshuffle('x', 0, 1) for r in \
                    read_samples_steps])
            read_pre_weights = TT.concatenate([rwp.dimshuffle('x', 0, 1) \
                    if rwp.ndim == 2 else TT.addbroadcast(rwp, 0) \
                    for rwp in read_ws_pre_steps]) if self.smoothed_diff_weights else None
        else:
            read_weights = read_weights_steps[0]
            read_samples = read_samples_steps[0]
            read_pre_weights = read_ws_pre_steps[0] if self.smoothed_diff_weights else None

        return read_weights, read_samples, h_t, mem_read, read_pre_weights

    def __step(self,
               state_below=None,
               mask=None,
               state_before=None,
               mem_before=None,
               mem_read_before=None,
               write_weight_before=None,
               read_weight_before=None,
               write_weight_before_pre=None,
               read_weight_before_pre=None,
               write_weight_samples=None,
               read_weight_samples=None,
               time_idxs=None,
               context=None,
               **kwargs):

        h_t = self.ff_controller.fprop(state_below=state_below,
                                       mem_before=mem_read_before,
                                       context=context)

        # First let's write the contents into the memory :
        write_weights, write_weights_samples, m_t, \
                write_heads_w, write_weights_pre = self.__get_writer_weights(h_t,
                                                                             mem_before=mem_before,
                                                                             state_below=state_below,
                                                                             write_weight_before=write_weight_before,
                                                                             write_weight_before_pre=write_weight_before_pre,
                                                                             time_idxs=time_idxs)

        # Now it is time to read :
        read_weights, read_weights_samples, h_t, mem_read_t, read_weights_pre = \
                self.__get_reader_weights(h_t,
                                          state_below=state_below,
                                          mem_before=m_t,
                                          write_heads_w=write_heads_w,
                                          read_weight_before=read_weight_before,
                                          read_weight_before_pre=read_weight_before_pre,
                                          time_idxs=time_idxs)

        if mask is not None:
            if m_t.ndim == 3:
                mask_ = mask.dimshuffle(0, 'x', 'x')
            else:
                mask_ = mask.dimshuffle(0, 'x')

            m_t = (1. - mask_) * mem_before + mask_ * m_t

            if h_t.ndim == 2:
                mask = mask.dimshuffle(0, 'x')
                if self.n_write_heads > 1:
                    maskw = mask.dimshuffle('x', 0, 'x')
                else:
                    maskw = mask

            mem_read_t = (1 - mask) * mem_read_before + mask * mem_read_t
            write_weights = write_weights.reshape(write_weight_before.shape)
            write_weights = (1 - maskw) * write_weight_before + \
                maskw * write_weights

            ret_vals = [h_t,
                        m_t,
                        mem_read_t,
                        write_weights]

            if h_t.ndim == 2:
                mask = mask.dimshuffle(0, 'x')
                if self.n_read_heads > 1:
                    maskr = mask.dimshuffle('x', 0, 'x')
                else:
                    maskr = mask

            read_weights = read_weights.reshape(read_weight_before.shape)
            read_weights = (
                1 - maskr) * read_weight_before + maskr * read_weights
            ret_vals.extend([ read_weights ])

            if self.smoothed_diff_weights:
                if write_weights_pre.ndim != write_weight_before_pre.ndim:
                    write_weights_pre = write_weights_pre.reshape(write_weight_before_pre.shape)

                write_weights_pre = (1 - maskw) * write_weight_before_pre + maskw * \
                        write_weights_pre

                ret_vals += [ write_weights_pre ]
                if read_weights_pre.ndim != read_weight_before_pre.ndim:
                    read_weights_pre = read_weights_pre.reshape(read_weight_before_pre.shape)

                read_weights_pre = (1 - maskr) * read_weight_before_pre + maskr * \
                        read_weights_pre
                ret_vals += [ read_weights_pre ]

            if self.use_reinforce:
                write_weights_samples = maskw * write_weights_samples
                ret_vals += [ write_weights_samples ]
                read_weights_samples = maskr * read_weights_samples
                ret_vals += [ read_weights_samples ]
        else:
            raise ValueError("mask should not be empty.")

        # Order here is important!
        return ret_vals

    def fprop(self,
              inp,
              context=None,
              mask=None,
              batch_size=None,
              cmask=None,
              use_mask=False,
              use_noise=False):

        use_mask = 0 if mask is None else 1
        if batch_size is not None:
            self.batch_size = batch_size

        if self.use_context and context is None:
            raise ValueError("Context should not be empty.")

        if not self.outputs_info:
            self.__create_states(inp)

        # This is to zero out the embedding where we provide
        # the target at the output layer.
        if cmask is not None:
            if mask.ndim == cmask.ndim:
                m = mask * TT.eq(cmask, 0).reshape((cmask.shape[0] * cmask.shape[1], -1))
            else:
                m = (mask.dimshuffle(0, 1, 'x') * TT.eq(cmask, 0))[:, :, 0].reshape((mask.shape[0] * mask.shape[1], -1))
        else:
            raise ValueError("Mask for the answers should not be empty.")

        if not self.use_bow_input:
            m = m.dimshuffle(0, 1, 'x')
            out = self.controller_inp.fprop(inp,
                                            deterministic=not use_noise)
            state_below = m * out.reshape((m.shape[0], m.shape[1], -1))
        else:
            m = m.dimshuffle(0, 1, 'x')
            out = self.controller_inp.fprop(inp,
                                            deterministic=not use_noise)
            state_below = m * out.reshape((inp.shape[0], inp.shape[1], -1))

        context_p = None
        if self.use_context:
            context_p = self.context_proj.fprop(context)

        def step_callback(*args):
            def lst_to_dict(lst):
                return {p.name: p for p in lst}

            state_below = args[0]
            if self.use_context:
                context = args[8]
                idx = 9
            else:
                context = None
                idx = 8

            if self.use_reinforce:
                if use_mask:
                    m = args[1]
                    step_res = self.__step(state_below=state_below,
                                           mask=m,
                                           mem_before=args[2],
                                           mem_read_before=args[3],
                                           write_weight_before=args[4],
                                           read_weight_before=args[5],
                                           write_weight_before_pre=args[6],
                                           read_weight_before_pre=args[7],
                                           context=context,
                                           time_idxs=args[idx],
                                           **lst_to_dict(args[idx+1:]))
                else:
                    step_res = self.__step(state_below=state_below,
                                           mem_before=args[1],
                                           mem_read_before=args[2],
                                           write_weight_before=args[3],
                                           read_weight_before=args[4],
                                           write_weight_before_pre=args[6],
                                           read_weight_before_pre=args[7],
                                           context=context,
                                           time_idxs=args[idx],
                                           **lst_to_dict(args[idx+1:]))
            else:
                if use_mask:
                    m = args[1]
                    step_res = self.__step(state_below=state_below,
                                           mask=m,
                                           mem_before=args[2],
                                           mem_read_before=args[3],
                                           write_weight_before=args[4],
                                           read_weight_before=args[5],
                                           write_weight_before_pre=args[6],
                                           read_weight_before_pre=args[7],
                                           context=context,
                                           time_idxs=args[idx],
                                           **lst_to_dict(args[idx+1:]))
                else:
                    step_res = self.__step(state_below=state_below,
                                           mem_before=args[1],
                                           mem_read_before=args[2],
                                           write_weight_before=args[3],
                                           read_weight_before=args[4],
                                           write_weight_before_pre=args[6],
                                           read_weight_before_pre=args[7],
                                           context=context,
                                           time_idxs=args[idx],
                                           **lst_to_dict(args[idx+1:]))
            return step_res

        seqs = [state_below]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.reshape((mask.shape[0], -1))
            mask = mask.dimshuffle(0, 1, 'x')
            seqs += [mask]

        if self.l1_pen and self.l1_pen > 0.:
            reg = abs(self.memory.address_vectors).sum()
            self.reg += self.l1_pen * reg

        if inp.ndim == 3:
            if not self.use_bow_input:
                seqs[:-1] = map(lambda x: x.reshape((inp.shape[0],
                                                     inp.shape[1],
                                                     -1)), seqs[:-1])
            else:
                seqs[:-1] = map(lambda x: x.reshape((inp.shape[0],
                                                     inp.shape[1],
                                                     -1)), seqs[:-1])
        else:
            seqs = map(lambda x: x.reshape((inp.shape[0],
                                            -1)), seqs)
        if self.seq_len is None:
            n_steps = inp.shape[0]
        else:
            n_steps = self.seq_len

        time_idxs = theano.shared(as_floatX(np.arange(self.mem_nel)), name="time_idxs")
        time_idxs = time_idxs.dimshuffle('x', 0)

        if self.use_context:
            non_sequences = [context_p, time_idxs] + self.params.values
        else:
            non_sequences = [time_idxs] + self.params.values

        # import ipdb; ipdb.set_trace()
        rval, updates = theano.scan(step_callback,
                                    sequences=seqs,
                                    outputs_info=self.outputs_info,
                                    n_steps=n_steps,
                                    non_sequences=non_sequences,
                                    strict=True)

        self.updates = updates
        return rval


class NTM(NTMBase):
    """
       A simple Neural Turing Machine implementation.
    """
    def __init__(self,
                 n_in=None,
                 n_hids=None,
                 use_loc_based_addressing=False,
                 use_multiscale_shifts=True,
                 use_reinforce=False,
                 use_gru_inp_rep=False,
                 smoothed_diff_weights=False,
                 mem_nel=None,
                 learn_h0=False,
                 n_reading_steps=1,
                 mem_size=None,
                 use_context=False,
                 n_layers=1,
                 hybrid_att=False,
                 weight_initializer=None,
                 bias_initializer=None,
                 controller_activ=None,
                 use_lstm_controller=False,
                 use_bow_input=True,
                 use_inp_content=True,
                 use_nogru_mem2q=False,
                 mem_gater_activ=None,
                 use_layer_norm=False,
                 recurrent_dropout_prob=0,
                 address_size=None,
                 erase_activ=None,
                 use_adv_indexing=False,
                 evaluation_mode=False,
                 use_soft_att=False,
                 use_hard_att_eval=False,
                 wpenalty=None,
                 noise=None,
                 use_quad_interactions=False,
                 l1_pen=None,
                 n_read_heads=1,
                 n_write_heads=1,
                 sampling=False,
                 batch_size=32,
                 content_activ=None,
                 seq_len=None,
                 use_noise=False,
                 dice_val=None,
                 name="gru_nmt"):

        if n_in is None:
            raise ValueError("Number of inputs should not be empty.")

        if n_read_heads is None or n_read_heads <= 0:
            raise ValueError("Number of read heads should be positive and "\
                             " it should not be None.")

        if n_write_heads is None or n_write_heads <= 0:
            raise ValueError("Number of write heads should be positive and "\
                             " it should not be None.")

        self.hybrid_att = hybrid_att
        if hybrid_att:
            if dice_val:
                self.dice = global_trng.binomial((1,), p=dice_val, n=1, dtype="float32").sum()
            else:
                self.dice = global_trng.binomial((1,), p=0.5, n=1, dtype="float32").sum()

        self.use_layer_norm = use_layer_norm
        self.recurrent_dropout_prob = recurrent_dropout_prob

        self.use_context = use_context
        self.wpenalty = wpenalty
        self.noise = noise
        self.use_gru_inp_rep = use_gru_inp_rep
        self.use_multiscale_shifts = use_multiscale_shifts
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.use_loc_based_addressing = use_loc_based_addressing
        self.use_reinforce = use_reinforce
        self.use_nogru_mem2q = use_nogru_mem2q
        self.n_reading_steps = n_reading_steps
        self.use_quad_interactions = use_quad_interactions
        self.smoothed_diff_weights = smoothed_diff_weights

        # Warning: The deep controller can not be used at the moment in this class.
        self.n_layers = n_layers
        self.use_soft_att = use_soft_att

        self.use_hard_att_eval = use_hard_att_eval

        if controller_activ is not None and isinstance(controller_activ, str):
            self.controller_activ = eval(controller_activ)
        elif controller_activ:
            self.controller_activ = controller_activ
        else:
            self.controller_activ = Tanh

        if mem_gater_activ is not None and isinstance(mem_gater_activ, str):
            self.mem_gater_activ = eval(mem_gater_activ)
        elif mem_gater_activ is not None:
            self.mem_gater_activ = mem_gater_activ
        else:
            self.mem_gater_activ = Sigmoid

        self.updates = OrderedDict()

        super(NTM, self).__init__(batch_size=batch_size,
                                  learn_h0=learn_h0,
                                  evaluation_mode=evaluation_mode)

        self.n_in = n_in
        self.n_hids = n_hids
        self.mem_nel = mem_nel
        self.mem_size = mem_size
        self.use_noise = use_noise
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.address_size = address_size if address_size else 0
        self.l1_pen = l1_pen
        self.use_inp_content = use_inp_content
        self.sampling = sampling
        self.use_adv_indexing = use_adv_indexing
        self.use_bow_input = use_bow_input
        self.use_lstm_controller = use_lstm_controller
        self.updates = OrderedDict()

        logger.info("Size of the address is %d" % self.address_size)
        logger.info("Learning the first state %d " % self.learn_h0)

        self.name = name
        self.outputs_info = []

        self.content_activ = content_activ or Trect
        self.erase_activ = erase_activ or Trect

        self.seq_len = seq_len
        self.init_params()
        self.__create_states()

    def __create_states(self, inp=None):
        if inp:
            bs = inp.shape[1]
            if inp.ndim == 4:
                bs = inp.shape[2]
            self.batch_size = bs
        else:
            if self.batch_size:
                bs = self.batch_size
            else:
                raise ValueError("Batch size is unspecified!")

        if not self.sampling:
            init_state = self._new_state((bs, self.n_hids),
                                          name=self.pname("init_state"))

            mem_state = TT.alloc(self.memory.M.dimshuffle('x', 0, 1),
                                 bs,
                                 self.memory.M.shape[0],
                                 self.memory.M.shape[1])

            mem_read = self._new_state((bs, self.mem_size + self.address_size),
                                       name=self.pname("mem_read"),
                                       adapt_state=False)

            if self.n_write_heads > 1:
                write_weights = self._new_state((self.n_write_heads,
                                                bs,
                                                self.mem_nel),
                                                name=self.pname("write_weights"),
                                                adapt_state=False)
            else:
                write_weights = self._new_state((bs, self.mem_nel),
                                                name=self.pname("write_weights"),
                                                adapt_state=False)

            self.outputs_info.extend([init_state, mem_state, mem_read, write_weights])

            if self.n_read_heads > 1:
                read_weights = self._new_state((self.n_read_heads,
                                                bs,
                                                self.mem_nel),
                                                name=self.pname("read_weights"),
                                                adapt_state=False)
            else:
                if self.n_reading_steps > 1:
                    read_weights = self._new_state((self.n_reading_steps,
                                                    bs,
                                                    self.mem_nel),
                                                    name=self.pname("read_weights"),
                                                    adapt_state=False)
                else:
                    read_weights = self._new_state((bs,
                                                    self.mem_nel),
                                                    name=self.pname("read_weights"),
                                                    adapt_state=False)

            self.outputs_info.append(read_weights)
        else:
            init_state = self._new_state((self.n_hids,), name=self.pname("init_state"))
            mem_state = self.memory.M
            mem_read = self._new_state((self.mem_size + self.address_size),
                                        name=self.pname("mem_read"),
                                        adapt_state=False)

            if self.n_write_heads > 1:
                write_weights = self._new_state((self.n_write_heads, self.mem_nel),
                                                 name=self.pname("write_weights"),
                                                 adapt_state=False)
            else:
                write_weights = self._new_state((self.mem_nel),
                                                 name=self.pname("write_weights"),
                                                 adapt_state=False)

            self.outputs_info.extend([init_state, mem_state, mem_read, \
                                        write_weights])

            if self.n_read_heads > 1:
                read_weights = self._new_state((self.n_read_heads, self.mem_nel),
                                                name=self.pname("read_weights"),
                                                adapt_state=False)
            else:
                read_weights = self._new_state((self.mem_nel,),
                                                name=self.pname("read_weights"),
                                                adapt_state=False)
            self.outputs_info.append(read_weights)

        if self.smoothed_diff_weights:
            if self.n_read_heads > 1:
                self.read_weights_pre = self._new_state((self.n_read_heads,
                                                         bs,
                                                         self.mem_nel),
                                                         name=self.pname("read_weights_pre"),
                                                         adapt_state=True)
            else:
                if self.n_reading_steps > 1:
                    self.read_weights_pre = self._new_state((self.n_reading_steps,
                                                             bs,
                                                             self.mem_nel),
                                                             name=self.pname("read_weights_pre"),
                                                             adapt_state=True)
                else:
                    self.read_weights_pre = self._new_state((bs,
                                                             self.mem_nel),
                                                             name=self.pname("read_weights_pre"),
                                                             adapt_state=True)

            if self.n_write_heads > 1:
                self.write_weights_pre = self._new_state((self.n_write_heads,
                                                          bs,
                                                          self.mem_nel),
                                                          name=self.pname("write_weights_pre"),
                                                          adapt_state=False)
            else:
                self.write_weights_pre = self._new_state((bs, self.mem_nel),
                                                         name=self.pname("write_weights_pre"),
                                                         adapt_state=False)

            self.outputs_info.extend([self.write_weights_pre, self.read_weights_pre])

        if self.use_reinforce:
            self.outputs_info.append(None)
            self.outputs_info.append(None)

    def init_params(self):
        if self.use_lstm_controller:
            cnames = [ "forget_below",
                       "input_below",
                       "output_below",
                       "state_below" ]

            self.controller = LSTMController(n_hids=self.n_hids,
                                             mem_size=self.mem_size + self.address_size,
                                             weight_initializer=self.weight_initializer,
                                             bias_initializer=self.bias_initializer,
                                             activ=self.controller_activ,
                                             name=self.pname("lstm_controller"))
        else:
            cnames = ["reset_below",
                      "gater_below",
                      "state_below"]

            self.controller = Controller(n_hids=self.n_hids,
                                         mem_size=self.mem_size + self.address_size,
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         activ=self.controller_activ,
                                         use_layer_norm=self.use_layer_norm,
                                         recurrent_dropout_prob=self.recurrent_dropout_prob,
                                         name=self.pname("controller"))

        self.erase_heads = [AffineLayer(n_in=self.n_hids,
                                        n_out=self.mem_size,
                                        weight_initializer=self.weight_initializer,
                                        bias_initializer=self.bias_initializer,
                                        name=self.pname("erase_head_ht_0"))]

        self.content_heads = [AffineLayer(n_in=self.n_hids,
                                          n_out=self.mem_size,
                                          weight_initializer=self.weight_initializer,
                                          bias_initializer=self.bias_initializer,
                                          name=self.pname("content_head_0"))]

        if self.use_quad_interactions and self.use_inp_content:
            logger.info("Using the quadratic interactions for the gating of the input.")
            self.quad_interact_layer = QuadraticInteractionLayer(n_in=self.n_in,
                                                                 n_out=self.n_hids,
                                                                 weight_initializer=self.weight_initializer,
                                                                 name=self.pname("quad_int_layer"))

        self.write_heads = [Addresser(n_hids=self.n_hids,
                                     mem_size=self.mem_size, #+ self.address_size,
                                     address_size=self.address_size,
                                     use_loc_based_addressing=self.use_loc_based_addressing,
                                     mem_nel=self.mem_nel,
                                     use_multiscale_shifts=self.use_multiscale_shifts,
                                     use_adv_indexing=self.use_adv_indexing,
                                     smoothed_diff_weights=self.smoothed_diff_weights,
                                     n_mid_key_size=self.n_hids,
                                     weight_initializer=self.weight_initializer,
                                     bias_initializer=self.bias_initializer,
                                     name=self.pname("addresser_head_0"))]

        for i in xrange(1, self.n_write_heads):
            self.write_heads.append(Addresser(n_hids=self.n_hids,
                                             mem_size=self.mem_size,
                                             mem_nel=self.mem_nel,
                                             smoothed_diff_weights=self.smoothed_diff_weights,
                                             address_size=self.address_size,
                                             use_loc_based_addressing=self.use_loc_based_addressing,
                                             use_adv_indexing=self.use_adv_indexing,
                                             use_multiscale_shifts=self.use_multiscale_shifts,
                                             n_mid_key_size=self.n_hids,
                                             weight_initializer=self.weight_initializer,
                                             bias_initializer=self.bias_initializer,
                                             name=self.pname("addresser_head_%d" % i)))

            self.erase_heads.append(AffineLayer(n_in=self.n_hids,
                                                n_out=self.mem_size,
                                                weight_initializer=self.weight_initializer,
                                                bias_initializer=self.bias_initializer,
                                                name=self.pname("erase_head_ht_%d" % i)))

            self.content_heads.append(AffineLayer(n_in=self.n_hids,
                                                  n_out=self.mem_size,
                                                  weight_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  name=self.pname("content_head_%d" % i)))


        nfout = len(cnames)
        self.cnames = map(lambda x: self.pname(x), cnames)
        self.controller_inps = ForkLayer(n_in=self.n_in,
                                         n_outs=[self.n_hids for _ in xrange(nfout)],
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         use_bow_input=False,
                                         wpenalty=self.wpenalty,
                                         noise=self.noise,
                                         use_bias=False,
                                         names=self.cnames)

        self.children = [self.controller, self.controller_inps] + self.write_heads + \
                         self.erase_heads + self.content_heads

        if self.use_quad_interactions and self.use_inp_content:
            self.children.append(self.quad_interact_layer)

        if self.use_inp_content:
            self.inp2_content = AffineLayer(n_in=self.n_hids,
                                            n_out=self.mem_size,
                                            weight_initializer=self.weight_initializer,
                                            bias_initializer=self.bias_initializer,
                                            name=self.pname("inp2_content"))

            self.inp_scale_content = AffineLayer(n_in=self.n_hids,
                                                 n_out=1,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 name=self.pname("inp_scale_2content"))

            self.children.extend([self.inp2_content, self.inp_scale_content])

        self.read_heads = [Addresser(n_hids=self.n_hids,
                                     mem_size=self.mem_size,
                                     mem_nel=self.mem_nel,
                                     smoothed_diff_weights=self.smoothed_diff_weights,
                                     address_size=self.address_size,
                                     n_mid_key_size=self.n_hids,
                                     use_loc_based_addressing=self.use_loc_based_addressing,
                                     use_adv_indexing=self.use_adv_indexing,
                                     use_multiscale_shifts=self.use_multiscale_shifts,
                                     weight_initializer=self.weight_initializer,
                                     bias_initializer=self.bias_initializer,
                                     name=self.pname("read_addresser_head_0"))]

        for i in xrange(1, self.n_read_heads):
            self.read_heads.append(Addresser(n_hids=self.n_hids,
                                             mem_size=self.mem_size,
                                             mem_nel=self.mem_nel,
                                             smoothed_diff_weights=self.smoothed_diff_weights,
                                             n_mid_key_size=self.n_hids,
                                             address_size=self.address_size,
                                             use_loc_based_addressing=self.use_loc_based_addressing,
                                             use_adv_indexing=self.use_adv_indexing,
                                             use_multiscale_shifts=self.use_multiscale_shifts,
                                             weight_initializer=self.weight_initializer,
                                             bias_initializer=self.bias_initializer,
                                             name=self.pname("read_addresser_head_%d" % i)))
        self.children += self.read_heads
        self.memory = AddressedMemory(mem_nel=self.mem_nel,
                                      mem_size=self.mem_size,
                                      address_size=self.address_size,
                                      n_read_heads=self.n_read_heads,
                                      n_write_heads=self.n_write_heads,
                                      n_reading_steps=self.n_reading_steps)

        self.children += [self.memory]
        if self.use_context:
            self.context_proj = AffineLayer(n_in=self.n_in,
                                            n_out=self.n_hids,
                                            weight_initializer=self.weight_initializer,
                                            bias_initializer=self.bias_initializer,
                                            name=self.pname("context_proj"))
            self.children += [self.context_proj]

        self.merge_params()

    def __get_writer_weights(self,
                             h_t=None,
                             h_t_below=None,
                             state_below=None,
                             write_weight_before=None,
                             write_weight_before_pre=None,
                             mem_before=None,
                             time_idxs=None):

        first_write_weight = write_weight_before[0] if self.n_write_heads > 1 \
                else write_weight_before

        first_write_weight_pre = write_weight_before_pre[0] if self.n_write_heads > 1 \
                else write_weight_before_pre

        write_heads_w, write_ws_pre  = self.write_heads[0].fprop(h_t,
                                                   mem_before[:, 1:, :],
                                                   first_write_weight,
                                                   w_t_pre_before=first_write_weight_pre,
                                                   time_idxs=time_idxs)

        for i in xrange(1, self.n_write_heads):
            write_heads_w_tmp, write_ws_pre_tmp = \
                    self.write_heads[i].fprop(h_t,
                                              mem_before[:, 1:, :],
                                              write_weight_before[i],
                                              w_t_pre_before=first_write_weight_pre,
                                              time_idxs=time_idxs)

            write_heads_w.extend(write_heads_w_tmp)
            write_ws_pre.extend(write_ws_pre_tmp)

        # TODO: change mem_read_before to mem_before
        erase_heads = [self.erase_activ(erase.fprop(h_t)) for erase in self.erase_heads]
        quad_interact = self.quad_interact_layer.fprop(h_t, h_t_below).dimshuffle(0, 'x') \
                if self.use_quad_interactions and self.use_inp_content else 0

        if self.use_inp_content:
            inp2_content = self.inp2_content.fprop(state_below)
            isc_inp = h_t + quad_interact \
                    if self.use_quad_interactions else h_t
            inp_scaler = self.inp_scale_content.fprop(isc_inp).reshape((\
                    inp2_content.shape[0], )).dimshuffle(0, 'x')
            inp_proj = inp2_content * Sigmoid(inp_scaler)
            contents = [self.content_activ(content.fprop(h_t) + inp_proj) \
                    for content in self.content_heads]
        else:
            contents = [self.content_activ(content.fprop(h_t)) for content in \
                    self.content_heads]

        if erase_heads[0].ndim == 2:
            erase_heads = [erase_head.dimshuffle(0, 'x', 1) for erase_head in \
                    erase_heads]

            write_heads_w = [w.dimshuffle(0, 1, 'x') for w in write_heads_w]

            if self.smoothed_diff_weights:
                write_ws_pre = TT.concatenate([wwp.dimshuffle('x', 0, 1) for wwp \
                        in write_ws_pre])

            contents = [content.dimshuffle(0, 'x', 1) for content in contents]

            if mem_before.ndim == 2:
                M_ = mem_before.dimshuffle('x', 0, 1)
            else:
                M_ = mem_before
        else:
            raise ValueError("erase_ht should be 2D!!!!")

        if not self.use_reinforce and self.evaluation_mode and self.use_hard_att_eval:
            print("Using discrete attention during the evaluation...")
            if isinstance(write_heads_w, list):
                write_heads_w = [ get_hard_vals(ww, tot_size=self.mem_nel) for ww in write_heads_w ]
            else:
                write_heads_w = get_hard_vals(write_heads_w, tot_size=self.mem_nel)

        write_weights_samples = write_heads_w

        if self.use_reinforce:
            write_weights_samples = [self._get_sample_weights(w) \
                                     for w in write_weights_samples]

        write_weights, write_weights_samples, m_t = self.memory.write(erase_heads, \
                write_weights_samples, write_heads_w, contents, M_)

        return write_weights, write_weights_samples, m_t, write_heads_w, write_ws_pre

    def __get_reader_weights_single_step(self,
                                         h_t=None,
                                         mem_before=None,
                                         write_heads_w=None,
                                         read_weight_before=None,
                                         read_samples_before=None,
                                         read_weight_before_pre=None,
                                         time_idxs=None):
        m_t = mem_before
        if self.n_read_heads > 1:
            first_read_weight = read_weight_before[0]
            first_read_weight_pre = read_weight_before_pre[0] if self.smoothed_diff_weights \
                    else None
        elif self.n_reading_steps > 1:
            first_read_weight = read_weight_before[-1]
            first_read_weight_pre = read_weight_before_pre[-1] if self.smoothed_diff_weights \
                    else None
        else:
            first_read_weight = read_weight_before
            first_read_weight_pre = read_weight_before_pre if self.smoothed_diff_weights \
                    else None

        read_heads_w, read_ws_pre = self.read_heads[0].fprop(h_t,
                                                             m_t[:, :-1, :],
                                                             first_read_weight,
                                                             w_t_pre_before=first_read_weight_pre,
                                                             time_idxs=time_idxs)

        for i in xrange(1, self.n_read_heads):
            read_heads_w_tmp, read_ws_pre_tmp = \
                    self.read_heads[i].fprop(h_t,
                                             m_t[:, :-1, :],
                                             read_weight_before[i],
                                             w_t_pre_before=read_weight_before_pre[i],
                                             time_idxs=time_idxs)

            read_heads_w.extend(read_heads_w_tmp)
            read_ws_pre.extend(read_ws_pre_tmp)

        if not self.use_reinforce and self.evaluation_mode and self.use_hard_att_eval:
            print("Using discrete attention during the evaluation...")
            if isinstance(read_heads_w, list):
                read_heads_w = [ get_hard_vals(rw, tot_size=self.mem_nel) for rw in read_heads_w ]
            else:
                read_heads_w = get_hard_vals(read_heads_w, tot_size=self.mem_nel)

        read_weights_samples = read_heads_w
        if self.use_reinforce:
            read_weights_samples = [self._get_sample_weights(w) \
                    for w in read_weights_samples]

        read_weights, read_weights_samples, mem_read_t = self.memory.read(read_heads_w, \
                read_weights_samples, m_t)

        if isinstance(read_ws_pre, list):
            read_ws_pre = TT.concatenate([r.dimshuffle(0, 1, 'x') if r.ndim == 2 else r\
                    for r in read_ws_pre]) if self.smoothed_diff_weights else None

        return read_weights, read_weights_samples, mem_read_t, read_ws_pre

    def __get_reader_weights(self,
                             h_t=None,
                             state_below=None,
                             reset_below=None,
                             gater_below=None,
                             mem_before=None,
                             write_heads_w=None,
                             read_weight_before=None,
                             read_weight_before_pre=None,
                             time_idxs=None):

        read_weights_steps = []
        read_samples_steps = []
        read_ws_pre_steps = []
        add_to_list = lambda x, y:  x.extend(y) if isinstance(y, list) else x.append(y)

        for i in xrange(self.n_reading_steps):
            read_weights, read_samples, mem_read, read_ws_pre = \
                self.__get_reader_weights_single_step(h_t,
                                                      mem_before=mem_before,
                                                      read_weight_before=read_weight_before,
                                                      read_weight_before_pre=read_weight_before_pre,
                                                      time_idxs=time_idxs)

            add_to_list(read_weights_steps, read_weights)
            add_to_list(read_samples_steps, read_samples)
            add_to_list(read_ws_pre_steps, read_ws_pre)

            if i < self.n_reading_steps - 1:
                read_weight_before = read_weights
                mem_read_before = mem_read
                h_t = self.controller.fprop(h_t,
                                            mem_read_before,
                                            reset_below,
                                            gater_below,
                                            state_below,
                                            use_noise=self.evaluation_mode)

        if self.n_reading_steps is not None and self.n_reading_steps > 1 or self.n_read_heads > 1:
            read_weights = TT.concatenate([r.dimshuffle('x', 0, 1) if r.ndim == 2 else r for r in \
                    read_weights_steps])
            read_samples = TT.concatenate([r.dimshuffle('x', 0, 1) if r.ndim == 2 else r for r in \
                    read_samples_steps])
            read_ws_pre = TT.concatenate([rwp.dimshuffle('x', 0, 1) if r.ndim == 2 else r for rwp in \
                    read_ws_pre_steps]) if self.smoothed_diff_weights else None
        else:
            read_weights = read_weights_steps[0]
            read_samples = read_samples_steps[0]
            read_ws_pre = read_ws_pre_steps[0] if self.smoothed_diff_weights else None

        return read_weights, read_samples, h_t, mem_read, read_ws_pre

    def __step(self,
               reset_below=None,
               gater_below=None,
               state_below=None,
               h_t_below=None,
               mask=None,
               state_before=None,
               mem_before=None,
               qmask=None,
               context=None,
               mem_read_before=None,
               write_weight_before=None,
               read_weight_before=None,
               write_weights_before_pre=None,
               read_weights_before_pre=None,
               write_weight_samples=None,
               read_weight_samples=None,
               time_idxs=None,
               **kwargs):

        prev_h = state_before
        if qmask and self.use_nogru_mem2q:
            prev_h = qmask * state_before

        h_t = self.controller.fprop(prev_h,
                                    mem_read_before,
                                    reset_below,
                                    gater_below,
                                    state_below,
                                    use_noise=self.evaluation_mode,
                                    context=None)

        # Writing related staff goes there:
        write_weights, write_weights_samples, m_t, write_heads_w, write_weights_pre = \
                self.__get_writer_weights(h_t,
                                          mem_before=mem_before,
                                          h_t_below=h_t_below,
                                          state_below=state_below,
                                          write_weight_before=write_weight_before,
                                          write_weight_before_pre=write_weights_before_pre,
                                          time_idxs=time_idxs)
        #Now it is time to read:
        read_weights, read_weights_samples, h_t, mem_read_t, read_weights_pre = \
                self.__get_reader_weights(h_t,
                                          state_below=state_below,
                                          reset_below=reset_below,
                                          gater_below=gater_below,
                                          mem_before=m_t,
                                          write_heads_w=write_heads_w,
                                          read_weight_before=read_weight_before,
                                          read_weight_before_pre=read_weights_before_pre,
                                          time_idxs=time_idxs)

        if read_weights.ndim != read_weight_before:
            read_weights = read_weights.reshape(read_weight_before.shape)

        if write_weights.ndim != write_weight_before.ndim:
            write_weights = write_weights.reshape(write_weight_before.shape)

        if mask is not None:
            if m_t.ndim == 3:
                if mask.ndim == 1:
                    mask_ = mask.dimshuffle(0, 'x', 'x')
                else:
                    mask_ = TT.addbroadcast(mask.dimshuffle(0, 1, 'x'), 1)
            else:
                mask_ = mask.dimshuffle(0, 'x')

            m_t = (1 - mask_) * mem_before + mask_ * m_t

            if h_t.ndim == 2:
                if mask.ndim == 1:
                    mask = mask.dimshuffle(0, 'x')
                else:
                    mask = TT.addbroadcast(mask, 1)
                if self.n_write_heads > 1:
                    maskw = mask.dimshuffle('x', 0, 'x')
                else:
                    maskw = mask

            h_t = (1 - mask) * state_before + mask * h_t
            mem_read_t = (1 - mask) * mem_read_before + mask * mem_read_t

            write_weights = (1 - maskw) * write_weight_before + \
                maskw * write_weights
            if h_t.ndim == 2:
                mask = mask.dimshuffle(0, 'x')
                if self.n_read_heads > 1:
                    maskr = mask.dimshuffle('x', 0, 'x')
                else:
                    maskr = mask

            ret_vals = [h_t, m_t,
                        mem_read_t,
                        write_weights]

            read_weights = (
                1 - maskr) * read_weight_before + maskr * read_weights
            ret_vals.extend([ read_weights ])

            if self.smoothed_diff_weights:
                if write_weights_pre.ndim != write_weights_before_pre.ndim:
                    write_weights_pre = write_weights_pre.reshape(write_weights_before_pre.shape)

                write_weights_pre = (1 - maskw) * write_weights_before_pre + maskw * \
                        write_weights_pre

                ret_vals += [ write_weights_pre ]

                if read_weights_pre.ndim != read_weights_before_pre.ndim:
                    read_weights_pre = read_weights_pre.reshape(read_weights_before_pre.shape)

                read_weights_pre = (1 - maskr) * read_weights_before_pre + maskr *\
                        read_weights_pre
                ret_vals += [ read_weights_pre ]

            if self.use_reinforce:
                write_weights_samples = maskw * write_weights_samples
                ret_vals += [ write_weights_samples ]
                read_weights_samples = maskr * read_weights_samples
                ret_vals += [ read_weights_samples ]
        else:
            if write_weights.ndim != write_weight_before.ndim:
                write_weights = write_weights.reshape(write_weight_before.shape)


            ret_vals = [h_t, m_t,
                        mem_read_t,
                        write_weights,
                        read_weights]

            if self.smoothed_diff_weights:
                if write_weights_pre.ndim != write_weights_before_pre.ndim:
                    write_weights_pre = write_weights_pre.reshape(write_weights_before_pre.shape)

                ret_vals += [write_weights_pre]

                if read_weights_pre.ndim != read_weights_before_pre.ndim:
                    read_weights_pre = read_weights_pre.reshape(read_weights_before_pre.shape)

                ret_vals += [read_weights_pre]

            if self.use_reinforce:
                ret_vals += [write_weights_samples, read_weights_samples]

        # Order here is important!
        return ret_vals

    def fprop(self,
              inp,
              mask=None,
              batch_size=None,
              cmask=None,
              context=None,
              use_mask=False,
              use_noise=False):

        if self.use_context and context is None:
            raise ValueError("Context should not be empty.")

        # Creating the initial states of the GRU controller.
        if not self.outputs_info:
            self.__create_states(inp)

        if batch_size is not None:
            self.batch_size = batch_size

        # This is to zero out the embedding where we
        # provide the target at the output layer.
        if use_mask:
            if cmask is not None:
                if mask.ndim == cmask.ndim:
                    m = (mask * TT.eq(cmask, 0)).reshape((cmask.shape[0] * cmask.shape[1], -1))
                else:
                    m = (mask.dimshuffle(0, 1, 'x') * TT.eq(cmask, 0))[:, :, 0].reshape((mask.shape[0] * mask.shape[1], -1))
            else:
                m = mask

        #import pdb; pdb.set_trace()
        qmask = None
        if use_mask:
            if self.use_nogru_mem2q:
                ncmask = cmask.sum(0)
                nmask = mask.sum(0)
                q_idx = nmask - ncmask - 1
                qmask = TT.set_subtensor(mask[TT.cast(q_idx, "int32"),
                                        TT.arange(q_idx.shape[0])], as_floatX(0))
                qmask = qmask.dimshuffle(0, 1, 'x')

            shp = (m.shape[0], m.shape[1], -1)
            #shp = (mask.shape[0], mask.shape[1], -1)
        else:
            shp = (inp.shape[0], inp.shape[1], -1)

        if (not self.use_bow_input or self.use_gru_inp_rep)  \
           and (self.use_bow_input or self.use_gru_inp_rep) and use_mask:
            m = m.dimshuffle(0, 1, 'x')

            outs = self.controller_inps.fprop(inp, deterministic=not use_noise)

            reset_below = (m * outs[self.cnames[0]].reshape(shp)).reshape((mask.shape[0],
                                                                           mask.shape[1],
                                                                           -1))
            gater_below = (m * outs[self.cnames[1]].reshape(shp)).reshape((mask.shape[0],
                                                                           mask.shape[1],
                                                                           -1))
            state_below = (m * outs[self.cnames[2]].reshape(shp)).reshape((mask.shape[0],
                                                                           mask.shape[1],
                                                                           -1))

        else:
            if not use_mask:
                m = None

            outs = self.controller_inps.fprop(inp,
                                              mask=m,
                                              deterministic=not use_noise)

            reset_below = outs[self.cnames[0]].reshape(shp)
            gater_below = outs[self.cnames[1]].reshape(shp)
            state_below = outs[self.cnames[2]].reshape(shp)

        def step_callback(*args):
            def lst_to_dict(lst):
                return {p.name: p for p in lst}

            reset_below, gater_below, state_below = args[0], args[1], args[2]
            h_t_below = None
            idx = 3

            if use_mask:
                m = args[idx]
                idx += 1
                qmask = None

                if self.use_nogru_mem2q:
                    qmask = args[idx]
                    idx += 1

                if self.use_quad_interactions:
                    h_t_below = args[idx]
                    idx += 1

                context = None
                if self.learn_h0 and self.use_context:
                    logger.info("Using the context.")
                    context_p = args[idx+7]
                    state_before = Tanh(context_p)
                    idx += 1
                    tidx = idx + 7 if self.smoothed_diff_weights else idx + 5
                elif self.use_context:
                    logger.info("Using the context 2.")
                    context = args[idx+7]
                    state_before = args[idx]
                    idx += 1
                    tidx = idx + 7 if self.smoothed_diff_weights else idx + 5
                else:
                    logger.info("Not using the context.")
                    state_before = args[idx]
                    idx += 1
                    tidx = idx + 6 if self.smoothed_diff_weights else idx + 4

                step_res = self.__step(reset_below=reset_below,
                                       gater_below=gater_below,
                                       state_below=state_below,
                                       h_t_below=h_t_below,
                                       mask=m,
                                       qmask=qmask,
                                       state_before=state_before,
                                       mem_before=args[idx],
                                       mem_read_before=args[idx+1],
                                       write_weight_before=args[idx+2],
                                       read_weight_before=args[idx+3],
                                       write_weights_before_pre=args[idx+4] if self.smoothed_diff_weights else None,
                                       read_weights_before_pre=args[idx+5] if self.smoothed_diff_weights else None,
                                       context=context,
                                       time_idxs=args[tidx],
                                       **lst_to_dict(args[tidx+1:]))

            else:
                idx = 3
                if self.use_quad_interactions:
                    h_t_below = args[idx]
                    idx += 1

                if self.learn_h0 and self.use_context:
                    context_p = args[idx+7]
                    state_before = Tanh(context_p)
                else:
                    state_before = args[idx]

                idx += 1

                step_res = self.__step(reset_below=reset_below,
                                       gater_below=gater_below,
                                       state_below=state_below,
                                       h_t_below=h_t_below,
                                       state_before=state_before,
                                       mem_before=args[idx],
                                       mem_read_before=args[idx+1],
                                       write_weight_before=args[idx+2],
                                       read_weight_before=args[idx+3],
                                       write_weights_before_pre=args[idx+4] if self.smoothed_diff_weights else None,
                                       read_weights_before_pre=args[idx+5] if self.smoothed_diff_weights else None,
                                       context=None,
                                       time_idxs=args[idx+6],
                                       **lst_to_dict(args[idx+8:]))
            return step_res

        seqs = [reset_below, gater_below, state_below]
        if use_mask:
            if mask is not None:
                if mask.ndim == 3:
                    mask = mask.reshape((mask.shape[0], -1))

                mask = mask.dimshuffle(0, 1, 'x')
                seqs += [mask]

        if self.l1_pen and self.l1_pen > 0.:
            reg = abs(self.memory.M[:, self.mem_size:]).sum()
            self.reg += self.l1_pen * reg

        """
        if inp.ndim == 3:
            if not self.use_bow_input or not self.use_gru_inp_rep:
                seqs[:-1] = map(lambda x: x.reshape(shp), seqs[:-1])
        else:
            seqs = map(lambda x: x.reshape(shp), seqs)
        """

        if self.use_nogru_mem2q:
            seqs += [qmask]

        if self.use_quad_interactions:
            seqs += [inp]

        if self.seq_len is None:
            n_steps = inp.shape[0]
        else:
            n_steps = self.seq_len

        time_idxs = theano.shared(as_floatX(np.arange(self.mem_nel)), name="time_idxs")
        time_idxs = time_idxs.dimshuffle('x', 0)

        if self.use_context:
            context_p = self.context_proj.fprop(block_gradient(context))
            non_sequences = [context_p, time_idxs] + self.params.values
        else:
            non_sequences = [time_idxs] + self.params.values

        rval, updates = theano.scan(step_callback,
                                    sequences=seqs,
                                    outputs_info=self.outputs_info,
                                    n_steps=n_steps,
                                    non_sequences=non_sequences,
                                    strict=True)

        self.updates = updates
        return rval

