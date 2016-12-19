import logging

import numpy as np
import theano
import theano.tensor as TT

from core.parameters import Parameters, BiasInitializer, BiasInitMethods
from core.commons import (Rect, Trect,
                                 Tanh, Sigmoid,
                                 EPS, global_trng)

from core.commons import floatX
from core.utils import safe_izip, concatenate, sample_weights_classic, \
                                const, as_floatX

from core.utils.nnet_utils import softmax3

from core.layers import (Layer, RecurrentLayer,
                                AffineLayer, ForkLayer,
                                PowerupLayer, MergeLayer)

from core.operators import MemorySimilarity, CircularConvolve, \
        CircularConvolveAdvIndexing, GeomEuclideanSigmoidDot

logger = logging.getLogger(__name__)
logger.disabled = False


class ContentBasedAddresser(Layer):
    pass


class Addresser(Layer):
    """
        An addressing Layer.
    """
    def __init__(self,
                 n_hids=None,
                 mem_size=None,
                 mem_nel=None,
                 address_size=None,
                 mem_gater_activ=None,
                 n_mid_key_size=None,
                 scale_size=None,
                 use_scale_layer=True,
                 smoothed_diff_weights=False,
                 use_local_att=False,
                 mem_weight_decay=0.96,
                 read_head=False,
                 use_loc_based_addressing=True,
                 shift_width=3,
                 scale_bias_coef=1.0,
                 use_adv_indexing=False,
                 use_multiscale_shifts=True,
                 use_geom_sig_dot=False,
                 use_reinforce=False,
                 weight_initializer=None,
                 bias_initializer=None,
                 name="nmt_addresser"):

        super(Addresser, self).__init__()

        self.n_hids = n_hids
        self.n_mid_key_size = n_mid_key_size
        self.mem_size = mem_size
        self.mem_nel = mem_nel
        self.use_reinforce = use_reinforce
        self.read_head = read_head
        self.scale_size = scale_size
        self.scale_bias_coef = scale_bias_coef
        self.address_size = address_size
        self.use_scale_layer = use_scale_layer
        self.use_adv_indexing = use_adv_indexing
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.use_loc_based_addressing = use_loc_based_addressing
        self.use_multiscale_shifts = use_multiscale_shifts
        self.shift_width = shift_width
        self.smoothed_diff_weights = smoothed_diff_weights
        self.mem_weight_decay = mem_weight_decay
        self.use_local_att = use_local_att

        if self.use_local_att:
            self.time_idxs = const(as_floatX(np.arange(self.mem_nel)))
            self.time_idxs.name = "time_idxs"

        if self.use_adv_indexing:
            print "Using the advanced indexing."
        else:
            print "Not using the advanced indexing."

        if mem_gater_activ:
            self.mem_gater_activ = mem_gater_activ
        else:
            self.mem_gater_activ = Sigmoid

        if use_geom_sig_dot:
            self.mem_similarity = GeomEuclideanSigmoidDot()
        else:
            self.mem_similarity = MemorySimilarity()

        self.init_params()

    def init_params(self):
        if not self.use_local_att:
            names = ["fork_state_beta_t",
                     "fork_state_key_t"]
            self.n_outs = [1, self.mem_size + self.address_size]

        else:
            names = ["fork_state_key_t"]
            self.n_outs = [self.mem_size + self.address_size]

        self.shift_size = self.mem_nel

        if self.use_multiscale_shifts:
            logger.info("Using the multiscale shifts.")
            if self.scale_size is None or self.scale_size < -1:
                self.scale_size = int(np.floor(np.log(self.mem_nel)))
                logger.info("Size of the scales is %d" % self.scale_size)

        self.shift_size = self.shift_width * self.scale_size

        binit_vals = [None, None]


        if self.smoothed_diff_weights:
            names.append("fork_state_diff_gate")
            self.n_outs += [1]
            binit_vals += [-0.16]

        if self.use_loc_based_addressing:
            names +=  [ "fork_state_gater_t",
                        "fork_state_shift_hat_t" ]

            self.n_outs += [1, self.shift_size]
            binit_vals += [None, None]

            if not self.use_reinforce:
                names += [ "fork_state_sharpen_hat_t" ]
                self.n_outs += [1]
                binit_vals += [0.001]

            if self.use_scale_layer:
                self.scale_layer = AffineLayer(n_in=self.n_hids,
                                               n_out=self.scale_size,
                                               weight_initializer=self.weight_initializer,
                                               bias_initializer=self.bias_initializer,
                                               use_bias=True,
                                               name=self.pname("scale_layer"))

                pname = self.scale_layer.params.getparamname("bias")
                arng = as_floatX(np.arange(self.scale_size))
                arng = arng / arng.sum()
                self.scale_layer.params[pname] = self.scale_bias_coef * arng
                self.children.extend([self.scale_layer])

        if self.use_local_att:
            bott_size = self.n_hids
            logger.info("Using the local attention.")
            self.state_below_local = AffineLayer(n_in=self.n_hids,
                                                 n_out=bott_size,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 use_bias=True,
                                                 name=self.pname("state_below_loc_layer"))

            self.weights_below_local = AffineLayer(n_in=self.mem_nel,
                                                   n_out=bott_size,
                                                   weight_initializer=self.weight_initializer,
                                                   bias_initializer=self.bias_initializer,
                                                   use_bias=False,
                                                   name=self.pname("weights_loc_layer"))

            self.mean_pred = AffineLayer(n_in=bott_size,
                                         n_out=1,
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         use_bias=True,
                                         name=self.pname("mean_pred"))

            self.children.extend([self.state_below_local, self.weights_below_local, self.mean_pred])

        names = map(lambda x: self.pname(x), names)
        self.names = names

        self.state_fork_layer = ForkLayer(n_in=self.n_hids,
                                          n_outs=self.n_outs,
                                          weight_initializer=self.weight_initializer,
                                          bias_initializer=self.bias_initializer,
                                          init_bias_vals = binit_vals,
                                          names=names)

        self.children.extend([self.state_fork_layer])
        self.powerup_layer = None
        self.merge_params()

    def fprop(self,
              state_below,
              memory,
              w_t_before,
              w_t_pre_before=None,
              time_idxs=None):

        if time_idxs is None:
            logger.info("Time indices are empty!")
            time_idxs = self.time_idxs

        fork_outs = self.state_fork_layer.fprop(state_below)
        idx = 0
        # First things first, content based addressing:
        if not self.use_local_att:
            beta_pre = fork_outs[self.names[0]]
            beta = TT.nnet.softplus(beta_pre).reshape((beta_pre.shape[0],))

            if (state_below.ndim != beta.ndim and beta.ndim == 2
                    and state_below.ndim == 3):
                beta = beta.reshape((state_below.shape[0], state_below.shape[1]))
            elif (state_below.ndim != beta.ndim and beta.ndim == 1
                    and state_below.ndim == 2):
                beta = beta.reshape((state_below.shape[0],))
            else:
                raise ValueError("Unknown shape for beta!")
            beta = TT.shape_padright(beta)
            idx = 1

        key_pre = fork_outs[self.names[idx]]
        idx += 1
        key_t = key_pre
        sim_vals = self.mem_similarity(key_t, memory)

        weights = sim_vals
        new_pre_weights = None

        if self.smoothed_diff_weights:
            dw_scaler = fork_outs[self.names[idx]]
            dw_scaler = TT.addbroadcast(dw_scaler, 1)
            weights = sim_vals - Sigmoid(dw_scaler) * w_t_pre_before
            new_pre_weights = self.mem_weight_decay * sim_vals + (1 - \
                    self.mem_weight_decay) * w_t_pre_before
            idx += 1
        std = 5

        """
        if self.use_local_att:
            mean = as_floatX(self.mem_nel) * Sigmoid(weights*self.mean_pred.fprop(state_below))
            exp_ws = -(time_idxs - mean)**2 / (2.0 * std)
            weights = exp_ws * weights
        """

        if self.use_local_att:
            w_tc = softmax3(weights) if weights.ndim == 3 else TT.nnet.softmax(weights)
        else:
            if weights.ndim == 3 and beta.ndim == 2:
                beta = beta.dimshuffle('x', 0, 1)
                w_tc = softmax3(weights * beta)
            else:
                # Content based weights:
                w_tc = TT.nnet.softmax(weights * beta)

        if self.use_local_att:
            first_loc_layer = Tanh(self.state_below_local.fprop(state_below) +\
                    self.weights_below_local.fprop(weights))
            mean = as_floatX(self.mem_nel) * Sigmoid(self.mean_pred.fprop(first_loc_layer))
            mean = TT.addbroadcast(mean, 1)
            exp_ws = TT.exp(-((time_idxs - mean)**2) / (2.0 * std))
            w_tc = exp_ws * w_tc
            w_tc = w_tc / w_tc.sum(axis=1, keepdims=True)

        if self.use_loc_based_addressing:
            # Location based addressing:
            g_t_pre = fork_outs[self.names[idx]]
            g_t = Sigmoid(g_t_pre).reshape((g_t_pre.shape[0],))

            if (state_below.ndim != g_t.ndim and g_t.ndim == 2
                    and state_below.ndim == 3):
                g_t = g_t.reshape((state_below.shape[0], state_below.shape[1]))
            elif (state_below.ndim != g_t.ndim and g_t.ndim == 1
                    and state_below.ndim == 2):
                g_t = g_t.reshape((state_below.shape[0],))
            else:
                raise ValueError("Unknown shape for g_t!")

            g_t = TT.shape_padright(g_t)
            w_tg = g_t * w_tc + (1 - g_t) * w_t_before
            shifts_pre = fork_outs[self.names[idx + 1]]

            if shifts_pre.ndim == 2:
                if self.use_multiscale_shifts:

                    if self.use_scale_layer:
                        scales = TT.exp(self.scale_layer.fprop(state_below))
                        scales = scales.dimshuffle(0, 'x', 1)
                    else:
                        scales = TT.exp(TT.arange(self.scale_size).dimshuffle('x', 'x', 0))

                    shifts_pre = shifts_pre.reshape((state_below.shape[0],
                                                     -1,
                                                     self.scale_size))

                    shifts_pre = (shifts_pre * scales).sum(-1)

                    if self.shift_width >= 0:
                        shifts_pre = shifts_pre.reshape((-1, self.shift_width, 1))

                elif self.shift_width >= 0:
                    shifts_pre = shifts_pre.reshape((-1, self.shift_width, 1))
                else:
                    shifts_pre = shifts_pre.reshape(
                        (state_below.shape[0], self.mem_nel))

                if state_below.ndim == 3:
                    shifts_pre = shifts_pre.dimshuffle(0, 1, 'x')
                    shifts_pre = shifts_pre - shifts_pre.max(1, keepdims=True).dimshuffle(0, 'x', 'x')
                else:
                    shifts_pre = shifts_pre.dimshuffle(0, 1)
                    shifts_pre = shifts_pre - shifts_pre.max(1, keepdims=True)
                    shifts_pre = shifts_pre.dimshuffle(0, 1, 'x')
            elif shifts_pre.ndim == 1:
                if self.use_multiscale_shifts:
                    if self.use_scale_layer:
                        scales = TT.exp(self.scale_layer.fprop(state_below))
                    else:
                        scales = TT.exp(TT.arange(self.scale_size))

                    shifts_pre = shifts_pre.reshape((-1, self.scale_size))
                    shifts_pre = (shifts_pre * scales).sum(-1)
                    if self.shift_width >= 0:
                        shifts_pre = shifts_pre.reshape((-1, self.shift_width, 1))
                    if self.shift_width >= 0:
                        shifts_pre = shifts_pre.reshape((-1, 1))
                elif self.shift_width >= 0:
                    shifts_pre = shifts_pre.reshape((-1, 1))
                else:
                    shifts_pre = shifts_pre.reshape((self.mem_nel,))

                if state_below.ndim == 2:
                    shifts_pre = TT.shape_padright(shifts_pre)
                    shifts_pre = shifts_pre - shifts_pre.max(0, keepdims=True)

            shifts = TT.exp(shifts_pre)
            if shifts.ndim == 2:
                shifts = shifts / shifts.sum(axis=0, keepdims=True)
            elif shifts.ndim == 3:
                shifts = shifts / shifts.sum(axis=1, keepdims=True)

            CC = CircularConvolveAdvIndexing if self.use_adv_indexing else\
                    CircularConvolve

            w_t_hat = CC()(weights=w_tg, shifts=shifts,
                           mem_size=self.mem_nel,
                           shift_width=self.shift_width)

            if self.use_reinforce:
                if w_t_hat.ndim == 2:
                    w_t = TT.nnet.softmax(w_t_hat)
                elif w_t_hat.ndim == 3:
                    w_t = softmax3(w_t_hat)
            else:
                gamma_pre = fork_outs[self.names[4]]
                assert w_t_hat.ndim == gamma_pre.ndim, ("The number of dimensions for "
                                                        " w_t_hat and gamma_pre should "
                                                        " be the same")

                if gamma_pre.ndim == 1:
                    gamma_pre = gamma_pre
                else:
                    gamma_pre = gamma_pre.reshape((gamma_pre.shape[0],))

                gamma_pre = TT.shape_padright(gamma_pre)
                gamma = TT.nnet.softplus(gamma_pre) + const(1)

                w_t = (abs(w_t_hat + const(1e-16))**gamma) + const(1e-42)
                if (state_below.ndim != shifts_pre.ndim and w_t.ndim == 2
                        and state_below.ndim == 3):
                    w_t = w_t.reshape((state_below.shape[0], state_below.shape[1]))
                    w_t = w_t.dimshuffle(0, 1, 'x')
                elif (state_below.ndim != w_t.ndim and w_t.ndim == 1
                        and state_below.ndim == 2):
                    w_t = w_t.reshape((state_below.shape[0],))
                    w_t = w_t.dimshuffle(0, 'x')

                if w_t.ndim == 2:
                    w_t = w_t / (w_t.sum(axis=-1, keepdims=True) + const(1e-6))
                elif w_t.ndim == 3:
                    w_t = w_t / (w_t.sum(axis=-1, keepdims=True) + const(1e-6))
        else:
            w_t = w_tc

        return [w_t], [new_pre_weights]


