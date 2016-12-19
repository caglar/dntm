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

from core.layers import (Layer, RecurrentLayer,
                                AffineLayer, ForkLayer,
                                PowerupLayer, MergeLayer)

from core.operators import MemorySimilarity, CircularConvolve, \
        CircularConvolveAdvIndexing, GeomEuclideanSigmoidDot
from core.ext.nunits import NTanhP

logger = logging.getLogger(__name__)
logger.disabled = False
np.random.seed(1234)


class Controller(Layer):
    """
    A Writer Layer.
    """
    def __init__(self,
                 n_hids=None,
                 mem_size=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 activ=None,
                 name="ntm_controller"):

        if isinstance(activ, str) and activ is not None:
            self.activ = eval(activ)
        elif activ is not None:
            self.activ = activ
        else:
            self.activ = Tanh

        super(Controller, self).__init__()

        self.n_hids = n_hids
        self.mem_size = mem_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.init_params()

    def init_params(self):
        self.state_gater_before_proj = AffineLayer(n_in=self.n_hids,
                                                   n_out=self.n_hids,
                                                   weight_initializer=self.weight_initializer,
                                                   bias_initializer=self.bias_initializer,
                                                   name=self.name + "_statebf_gater")

        self.state_reset_before_proj = AffineLayer(n_in=self.n_hids,
                                                   n_out=self.n_hids,
                                                   weight_initializer=self.weight_initializer,
                                                   bias_initializer=self.bias_initializer,
                                                   name=self.name + "_statebf_reset")

        self.state_mem_before_proj = AffineLayer(n_in=self.mem_size,
                                                 n_out=self.n_hids,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 use_bias=True,
                                                 name=self.name + "_membf_ht")

        self.state_str_before_proj = AffineLayer(n_in=self.n_hids,
                                                 n_out=self.n_hids,
                                                 use_bias=False,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 name=self.name + "_state_before_ht")

        self.children = [self.state_gater_before_proj, self.state_reset_before_proj,
                         self.state_mem_before_proj, self.state_str_before_proj]

        self.merge_params()
        self.str_params()

    def fprop(self, state_before, mem_before,
              reset_below, gater_below,
              state_below, context=None):

        state_reset = self.state_reset_before_proj.fprop(state_before)
        state_gater = self.state_gater_before_proj.fprop(state_before)
        reset = Sigmoid(reset_below + state_reset)
        state_state = self.state_str_before_proj.fprop(reset * state_before)
        membf_state = self.state_mem_before_proj.fprop(mem_before)

        gater = Sigmoid(gater_below + state_gater)
        if context:
            h = self.activ(state_state + membf_state + state_below + context)
        else:
            h = self.activ(state_state + membf_state + state_below)
        h_t = (1 - gater) * state_before + gater * h
        return h_t


class FFController(Layer):
    """
    A Writer Layer.
    """
    def __init__(self,
                 n_hids=None,
                 mem_size=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 activ=None,
                 noisy=False,
                 n_layers=2,
                 name="ntm_controller"):

        if isinstance(activ, str) and activ is not None:
            self.activ = eval(activ)
        elif activ is not None:
            self.activ = activ
        else:
            self.activ = Tanh

        super(FFController, self).__init__()

        print "Number of layers is, ", n_layers
        self.n_layers = n_layers
        self.n_hids = n_hids
        self.mem_size = mem_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.use_noise = 1.0
        self.noisy = noisy
        self.init_params()

    def init_params(self):
        self.additional_layers = []

        if self.n_layers > 1:
            for i in xrange(1, self.n_layers):
                self.additional_layers += [ AffineLayer(n_in=self.n_hids,
                                                        n_out=self.n_hids,
                                                        weight_initializer=self.weight_initializer,
                                                        bias_initializer=self.bias_initializer,
                                                        name=self.pname("ff_cont_proj_%d" % i))]

        if self.noisy:
            mpname = self.pname("ff_controller_p_vals")
            self.params[mpname] = np.random.uniform(-1.0, 1.0, (self.n_layers, self.n_hids)).astype("float32")
            self.pvals = self.params[mpname]

        self.mem_before_p = AffineLayer(n_in=self.mem_size,
                                        n_out=self.n_hids,
                                        weight_initializer=self.weight_initializer,
                                        bias_initializer=self.bias_initializer,
                                        name=self.pname("mem_before_p"))

        self.children = [self.mem_before_p] + self.additional_layers
        self.merge_params()
        self.str_params()

    def fprop(self,
              state_below,
              mem_before=None,
              context=None):

        mem_before_p = 0.

        if mem_before:
            mem_before_p = self.mem_before_p.fprop(mem_before)

        if context:
            z_t = state_below + mem_before_p + context
        else:
            z_t = state_below + mem_before_p

        #import ipdb; ipdb.set_trace()
        if self.n_layers > 1:
            for i in xrange(1, self.n_layers):
                if self.noisy:
                    z_t = NTanhP(z_t, self.pvals[i-1], use_noise=self.use_noise)
                else:
                    z_t = self.activ(z_t)
                z_t = self.additional_layers[i-1].fprop(z_t)

        if self.noisy:
            h_t = NTanhP(z_t, self.pvals[self.n_layers - 1], use_noise=self.use_noise)
        else:
            h_t = self.activ(z_t)

        return h_t


class LSTMController(Layer):
    """
    A Writer Layer.
    """
    def __init__(self,
                 n_hids=None,
                 mem_size=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 activ=None,
                 name="ntm_lstm_controller"):

        if isinstance(activ, str) and activ is not None:
            self.activ = eval(activ)
        elif activ is not None:
            self.activ = activ
        else:
            self.activ = Rect

        super(LSTMController, self).__init__()

        self.n_hids = n_hids
        self.mem_size = mem_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.init_params()

    def init_params(self):
        st_bf_names = ["forget_stbf", "input_stbf", "out_stbf", "cell_stbf"]
        mem_bf_names = ["forget_membf", "input_membf", "out_membf", "cell_membf"]

        self.sbf_names = map(lambda x: self.pname(x), st_bf_names)
        self.mbf_names = map(lambda x: self.pname(x), mem_bf_names)

        binit_vals = [-1e-5 for i in xrange(len(st_bf_names))]
        nouts = [self.n_hids for i in xrange(len(mem_bf_names))]

        self.state_before_fork_layer = ForkLayer(n_in=self.n_hids,
                                                 n_outs=nouts,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 init_bias_vals=binit_vals,
                                                 names=self.sbf_names)

        self.mem_before_fork_layer = ForkLayer(n_in=self.mem_size,
                                               n_outs=nouts,
                                               weight_initializer=self.weight_initializer,
                                               bias_initializer=self.bias_initializer,
                                               init_bias_vals=binit_vals,
                                               names=self.mbf_names)

        self.children = [self.state_before_fork_layer, self.mem_before_fork_layer]
        self.merge_params()
        self.str_params()

    def fprop(self,
              state_before,
              mem_before,
              cell_before,
              forget_below,
              input_below,
              output_below,
              state_below):

        state_fork_outs = self.state_before_fork_layer.fprop(state_before)
        mem_fork_outs = self.mem_before_fork_layer.fprop(mem_before)

        inp = Sigmoid(input_below + mem_fork_outs[self.mbf_names[1]] + \
                state_fork_outs[self.sbf_names[1]])

        output = Sigmoid(output_below + mem_fork_outs[self.mbf_names[2]] + \
                state_fork_outs[self.sbf_names[2]])

        forget = Sigmoid(forget_below + mem_fork_outs[self.mbf_names[0]] + \
                state_fork_outs[self.sbf_names[0]])

        cell = Tanh(state_below + mem_fork_outs[self.mbf_names[3]] +
                state_fork_outs[self.sbf_names[3]])

        c_t = inp * cell + forget * cell_before
        h_t = output * self.activ(c_t)

        return h_t, c_t
