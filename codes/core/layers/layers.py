import abc
from abc import ABCMeta

import logging
from collections import OrderedDict

import numpy

import theano
import theano.tensor as TT

from ..utils import dot, safe_izip, as_floatX, block_gradient, sharedX
from ..commons import Sigmoid, Tanh, Trect, Rect
from ..parameters import Parameters
from ..basic import Basic
from theano.sandbox.cuda.dnn import dnn_batch_normalization_train, \
        dnn_batch_normalization_test
logger = logging.getLogger(__name__)


class AbstractLayer:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def init_params(self):
        pass

    @abc.abstractmethod
    def fprop(self):
        pass


class Layer(AbstractLayer, Basic):
    """
    Simple Layer base class.
    """
    def use_params(self, params):
        if hasattr(self, "name"):
            if "ntm_model_ntm" in self.name:
                self.params.set_values(params.filterby(self.name) + params.filterby("Addressed"))
            else:
                self.params.set_values(params.filterby(self.name))
        elif hasattr(self, "names"):
            for i, name in enumerate(self.names):
                self.children[i].use_params(params.filterby(name))
            self.merge_params()

    def constrain_params(self, params):
        self.reg = sum(constraint.constrain_param(param) for param in params \
                for constraint in self.constraints)

    def __init__(self):
        self.reg = 0
        self.children = []
        self.params = Parameters()
        self.updates = {}


class BatchNormLayer(Layer):
    """
        Basic batch normalization layer.
    """
    def __init__(self, n_in, n_out, noise=None, gamma_init=0.1,
                 beta_init=1e-6,
                 alpha=0.1,
                 name=None):
        self.n_in = n_in
        self.n_out = n_out
        self.noise = noise
        self.alpha = alpha
        self.gamma_init = gamma_init
        self.beta_init = beta_init
        self.name = name
        self.scale_running_ave = sharedX(numpy.zeros((self.n_out,)) + self.gamma_init,
                                         name=self.pname("running_scale"))

        self.mean_running_ave = sharedX(numpy.zeros((self.n_out,)) + self.beta_init,
                                        name=self.pname("running_mean"))

        super(BatchNormLayer, self).__init__()

        self.init_params()
        dev = theano.config.device
        use_cudnn = (dev.startswith('cuda') or dev.startswith('gpu'))

        if not use_cudnn:
            raise ValueError("This class does not support the batch norm on CPU.")

    def init_params(self):
        gamma = numpy.zeros((self.n_out,)) + self.gamma_init
        beta = numpy.zeros((self.n_out,)) + self.beta_init
        self.params[self.pname("gamma")] = gamma
        self.params[self.pname("beta")] = beta

    def fprop(self, inp, inference=False):
        gamma = self.params.filterby("gamma").values[0].dimshuffle('x', 0)
        beta = self.params.filterby("beta").values[0].dimshuffle('x', 0)

        X = inp
        if inp.ndim > 2:
            X = inp.reshape((-1, inp.shape[-1]))

        if inference:
            mean = self.mean_running_ave.dimshuffle('x', 0)
            var = self.scale_running_ave.dimshuffle('x', 0)

            out = dnn_batch_normalization_test(X, gamma, beta, mean, var)
        else:
            out, mean, invstd = dnn_batch_normalization_train(X, gamma, beta)
            self.updates[self.scale_running_ave] = TT.cast((1 - self.alpha) * self.scale_running_ave + \
                    self.alpha * (1/invstd.flatten())**2, "float32")
            self.updates[self.mean_running_ave] = TT.cast((1 - self.alpha) * self.mean_running_ave + \
                    self.alpha * mean.flatten(), "float32")
        return out.reshape(inp.shape)


class RecurrentLayer(Layer):
    """
        Abstract recurrent base layer. This basically defines the main skeleton of a
    recurrent layer.
    """
    def __init__(self):
        super(RecurrentLayer, self).__init__()

    def __step(self):
        pass


class AffineLayer(Layer):
    """
    This layer basically just does a linear projection.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 use_bias=True,
                 noise=None,
                 wpenalty=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 init_bias_val=None,
                 name=None):

        self.n_in = n_in
        self.n_out = n_out
        self.init_bias_val = init_bias_val
        self.noise = noise
        self.wpenalty = wpenalty

        if name is None:
            raise ValueError("name should not be empty!")

        super(AffineLayer, self).__init__()
        self.use_bias = use_bias
        self.name = name
        self.bias_initializer = bias_initializer
        self.weight_initializer = weight_initializer
        self.init_params()

    def init_params(self):
        W = self.weight_initializer(self.n_in, self.n_out)
        self.params[self.pname("weight")] = W
        if self.use_bias:
            b = self.bias_initializer(self.n_out,
                                      init_bias_val=self.init_bias_val)
            self.params[self.pname("bias")] = b

    def fprop(self, inp, deterministic=True):
        weight = self.params.filterby("weight").values[0]

        if self.noise and deterministic:
            weight = self.noise(weight)

        if self.wpenalty:
            weight = self.wpenalty(weight)

        if self.n_in == 1:
            weight = weight.dimshuffle(0, 'x', 1)
            weight = TT.addbroadcast(weight, 0)
            out = inp.dimshuffle(0, 1, 'x') * weight
            if self.use_bias:
                out += self.params.filterby("bias").values[0]
        else:
            out = dot(inp, weight)

            if self.use_bias:
                out += self.params.filterby("bias").values[0]
        return out


class QuadraticInteractionLayer(Layer):
    """
    This layer basically implements the quadratic interaction between two variables,
    such as a^{\top} W b^{\top} where W is a square matrix.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 noise=None,
                 wpenalty=None,
                 weight_initializer=None,
                 name=None):

        self.n_in = n_in
        self.n_out = n_out
        self.noise = noise
        self.wpenalty = wpenalty

        if name is None:
            raise ValueError("name should not be empty!")

        super(QuadraticInteractionLayer, self).__init__()
        self.name = name
        self.weight_initializer = weight_initializer
        self.init_params()

    def init_params(self):
        W = self.weight_initializer(self.n_in, self.n_out)
        self.params[self.pname("weight")] = W

    def fprop(self, inp1, inp2, deterministic=True):
        weight = self.params.filterby("weight").values[0]

        if self.noise and deterministic:
            weight = self.noise(weight)

        if self.wpenalty:
            weight = self.wpenalty(weight)

        out1 = dot(inp2, weight)
        out2 = (inp1 * out1).sum(axis=-1)
        return out2


class BOWLayer(Layer):
    """
    This layer implements a basic BOW layer.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 seq_len=12,
                 noise=None,
                 wpenalty=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 init_bias_val=None,
                 use_average=False,
                 use_positional_encoding=True,
                 use_inv_cost_mask=True,
                 use_q_embed=False,
                 name=None):

        self.n_in = n_in
        self.n_out = n_out
        self.init_bias_val = init_bias_val
        self.noise = noise
        self.wpenalty = wpenalty
        self.use_positional_encoding = use_positional_encoding
        self.seq_len = seq_len
        self.use_q_embed = use_q_embed

        if name is None:
            raise ValueError("name should not be empty!")

        super(BOWLayer, self).__init__()
        self.use_inv_cost_mask = use_inv_cost_mask
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.use_average = use_average
        self.name = name
        self.init_params()

    def init_params(self):
        self.proj_layer = AffineLayer(self.n_in,
                                      self.n_out,
                                      use_bias=False,
                                      noise=self.noise,
                                      wpenalty=self.wpenalty,
                                      weight_initializer=self.weight_initializer,
                                      name=self.pname("bow_proj"))

        self.qproj_layer = AffineLayer(self.n_in,
                                       self.n_out,
                                       use_bias=False,
                                       noise=self.noise,
                                       wpenalty=self.wpenalty,
                                       weight_initializer=self.weight_initializer,
                                       name=self.pname("bow_qproj"))

        self.children = [self.proj_layer]

        if self.use_q_embed:
            self.children += [self.qproj_layer]

        self.merge_params()
        if self.use_positional_encoding:
            self.peW = self.weight_initializer(self.seq_len, self.n_out)
            self.params[self.pname("penc_W")] = self.peW
            self.peW = self.params[self.pname("penc_W")]

        if self.use_q_embed:
            self.qpeW = self.weight_initializer(self.seq_len, self.n_out)
            self.params[self.pname("qpenc_W")] = self.qpeW
            self.qpeW = self.params[self.pname("qpenc_W")]

    def fprop(self, inp, amask=None, qmask=None, deterministic=True):
        """
        This function assumes that the input is 3D or 2D:
            n_f: Number of facts + Number of Questions + delimiters
            n_w: Number of words at each fact
            n_bs: Batch_size
        so the dims should be:
            n_w x n_f x n_bs
        """
        if inp.ndim == 3:
            imask = TT.gt(inp, 0).dimshuffle(0, 1, 2, 'x')
            rshp_pattern = (inp.shape[0], inp.shape[1], inp.shape[2], -1)
            dimshuffle_pattern = (0, 'x', 'x', 1)
        elif inp.ndim == 2:
            imask = TT.gt(inp, 0).dimshuffle(0, 1, 'x')
            rshp_pattern = (inp.shape[0], inp.shape[1], -1)
            dimshuffle_pattern = (0, 'x', 1)
        else:
            raise ValueError("There is a problem with the dimensions of the input.")

        out = self.proj_layer.fprop(inp,
                                    deterministic=deterministic).reshape(rshp_pattern)

        if self.use_q_embed:
            qout = self.qproj_layer.fprop(inp,
                                          deterministic=deterministic).reshape(rshp_pattern)
            if qmask is not None:
                qmask = qmask.diimshuffle(dimshuffle_pattern)
                qout = qmask * qout

        if self.use_positional_encoding:
            out = imask * (out + self.peW.dimshuffle(dimshuffle_pattern))

        if (qmask is not None) ^ self.use_q_embed:
            qout = imask * (qout + self.qpeW.dimshuffle(0, 'x', 'x', 1))

        if self.use_average:
            bow = out.mean(0)
            if (qmask is not None) ^ self.use_q_embed:
                qbow = qout.mean(0)

            if self.use_inv_cost_mask:
                m = amask
                if m is not None and m.ndim != out.ndim:
                    m = m.reshape((out.shape[1], out.shape[2],)).dimshuffle(0, 1, 'x')
                    bow = m * bow

                if (qmask is not None) ^ self.use_q_embed:
                    qbow = m * qbow
                    qmask = qmask.dimshuffle(0, 1, 'x')

            if (qmask is None) ^ self.use_q_embed:
                bow = qmask * qbow + (1 - qmask) * bow

        else:
            bow = out.sum(0)
            if (qmask is not None) ^ self.use_q_embed:
                qbow = qout.sum(0)

            if self.use_inv_cost_mask:
                assert amask is not None
                m = amask
                if m.ndim != out.ndim:
                    m = m.reshape((out.shape[1], out.shape[2],)).dimshuffle(0, 1, 'x')

                bow = m * bow
                if (qmask is not None) ^ self.use_q_embed:
                    qbow = m * qbow
                    qmask = qmask.dimshuffle(0, 1, 'x')

            if (qmask is not None) ^ self.use_q_embed:
                bow = qmask * qbow + (1 - qmask) * bow

        if inp.ndim == 2:
            bow_shp = (inp.shape[1], -1)
        else:
            bow_shp = (inp.shape[1], inp.shape[2], -1)

        bow = bow.reshape(bow_shp)
        return bow


class DeepAffineLayer(Layer):
    """
    Deep Feedforward layer.
    """
    def __init__(self,
                 n_in,
                 n_hids=None,
                 activ=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 name=None):

        cname = self.__class__.__name__
        if name is None:
            raise ValueError("%s's name should not be empty!" % cname)

        if n_hids is None:
            raise ValueError("%s's number of hidden units should not be empty." % cname)

        if activ is None:
            raise ValueError("%s's activation function should not be None" % cname)

        if weight_initializer is None:
            raise ValueError("%s's weight_initializer should not be None" % cname)

        if bias_initializer is None:
            raise ValueError("%s's bias_initializer should not be None" % cname)

        super(DeepAffineLayer, self).__init__()

        self.n_in = n_in
        self.n_hids = n_hids
        self.name = name
        self.activ = activ
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.n_layers = len(n_hids)
        self.children = []
        self.init_params()

    def init_params(self):
        first_layer = AffineLayer(n_in=self.n_in,
                                  n_out=self.n_hids[0],
                                  use_bias=True,
                                  weight_initializer=self.weight_initializer,
                                  bias_initializer=self.bias_initializer,
                                  name = self.pname("deep_layer_0"))
        self.children = [first_layer]

        for i in xrange(1, self.n_layers):
            this_layer = AffineLayer(n_in=self.n_in,
                                     n_out=self.n_hids[i],
                                     weight_initializer=self.weight_initializer,
                                     bias_initializer=self.bias_initializer,
                                     name=self.pname("deep_layer_" + i ))
            self.children.append(this_layer)

        self.merge_params()

    def fprop(self, inp):
        out = self.children[0].fprop(inp)
        for i in xrange(1, len(self.children)):
            an_out = self.activ(out)
            out = self.children[i].fprop(an_out)
        return out


class PowerupLayer(Layer):
    """
    This layer basically just does a linear projection.
    """
    def __init__(self,
                 n_in,
                 n_pools,
                 power_initializer=None,
                 bias_initializer=None,
                 eps=1e-8,
                 name=None):

        self.n_in = n_in
        self.n_pools = n_pools
        self.eps = eps
        super(PowerupLayer, self).__init__()

        if n_pools == 0:
            raise ValueError("The number of pools can not be 0!")

        if not (n_in % n_pools == 0):
            raise ValueError("Number of pools can not evenly divide the number"
                             "of inputs for Poweruplayer.")

        self.n_out = n_in / n_pools
        if name is None:
            raise ValueError("name should not be empty!")

        self.name = name
        self.params = Parameters()
        self.bias_initializer = bias_initializer
        self.power_initializer = power_initializer
        self.init_params()

    def init_params(self):
        p = self.power_initializer(self.n_out)
        self.params[self.pname("Power")] = p
        c = self.bias_initializer(self.n_pools)
        self.params[self.pname("Center")] = c

    def fprop(self, inp):
        center = self.params.filterby("Center").values[0]
        power = TT.nnet.softplus(self.params.filterby("Power").values[0] + self.eps) + 1
        if inp.ndim == 3:
            inp = inp.reshape((inp.shape[0], inp.shape[1], self.n_pools, self.n_in / self.n_pools))
            center = center.dimshuffle('x', 'x', 0, 'x')
            pup = power.dimshuffle('x', 'x', 'x', 0)
            sum_axis = 2
            power_shuffle_pattern = ('x', 0, 'x')
        elif inp.ndim == 2:
            inp = inp.reshape((inp.shape[0], self.n_pools, self.n_in / self.n_pools))
            center = center.dimshuffle('x', 0, 'x')
            pup = power.dimshuffle('x', 'x', 0)
            sum_axis = 1
            power_shuffle_pattern = ('x', 0)

        out = abs(inp - center) + self.eps

        out = (out**pup + self.eps).sum(sum_axis)
        pdown = power.dimshuffle(power_shuffle_pattern)
        out = out**(1.0/pdown)

        return out


class ForkLayer(Layer):
    """
        Creates multiple outputs from a single input.
    """
    def __init__(self,
                 n_in,
                 n_outs,
                 use_bias=True,
                 noise=None,
                 wpenalty=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 init_bias_vals=None,
                 use_bow_input=False,
                 names=None):

        super(ForkLayer, self).__init__()

        self.n_in = n_in
        self.n_outs = n_outs
        self.use_bow_input = use_bow_input
        self.noise = noise
        self.wpenalty = wpenalty

        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = weight_initializer
        self.names = names

        if init_bias_vals:
            if not isinstance(init_bias_vals, list):
                self.init_bias_vals = [init_bias_vals for i in xrange(len(n_outs))]
            else:
                self.init_bias_vals = init_bias_vals
        else:
            self.init_bias_vals = [None for i in xrange(len(n_outs))]

        if not self.use_bow_input:
            self.children = [AffineLayer(n_in=n_in,
                                         n_out=numpy.sum(n_outs),
                                         use_bias=use_bias,
                                         noise=self.noise,
                                         wpenalty=self.wpenalty,
                                         weight_initializer=weight_initializer,
                                         bias_initializer=bias_initializer,
                                         init_bias_val=None,#self.init_bias_vals[i],
                                         name="_".join(names))]
        else:
            self.children = [BOWLayer(n_in=n_in,
                                      n_out=numpy.sum(n_outs),
                                      noise=self.noise,
                                      wpenalty=self.wpenalty,
                                      weight_initializer=weight_initializer,
                                      name="_".join(names[i]))]
        self.init_params()

    def init_params(self):
        self.params = sum(child.params for child in self.children)

    def slice(self, tensor, start, end):
        if tensor.ndim == 2:
            return tensor[:, start:end]
        elif tensor.ndim == 3:
            return tensor[:, :, start:end]

    def fprop(self, inp, mask=None, deterministic=True):
        outs = OrderedDict({})
        #import ipdb; ipdb.set_trace()
        if self.use_bow_input:
            tot_inp = self.children[0].fprop(inp,
                                             amask=mask,
                                             deterministic=deterministic)

            for i, name in enumerate(self.names):
                if i == 0:
                    outs[name] = self.slice(tot_inp, 0, self.n_outs[0])
                else:
                    start = numpy.sum(self.n_outs[:i])
                    outs[name] = self.slice(tot_inp, start,
                                            start + self.n_outs[i])
        else:
            tot_inp = self.children[0].fprop(inp, deterministic=deterministic)
            for i, name in enumerate(self.names):
                if i == 0:
                    outs[name] = self.slice(tot_inp, 0, self.n_outs[0])
                else:
                    start = numpy.sum(self.n_outs[:i])
                    outs[name] = self.slice(tot_inp, start,
                                            start + self.n_outs[i])
        return outs


class MergeLayer(Layer):
    """
        Takes multiple inputs and merge their input to get one output.
    """
    def __init__(self,
                 n_ins,
                 n_out,
                 use_bias=True,
                 weight_initializer=None,
                 bias_initializer=None,
                 names=None):

        super(MergeLayer, self).__init__()

        self.n_ins = n_ins
        self.n_out = n_out
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = weight_initializer
        self.names = names

        self.children = [ AffineLayer(n_in=n_ins[i],
                                     n_out=n_out,
                                     use_bias=use_bias,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer,
                                     name=names[i])
                                      for i in xrange(len(n_ins)) ]
        self.init_params()

    def init_params(self):
        self.params = sum(child.params for child in self.children)

    def fprop(self, inps):
        out = sum(child.fprop(inps[i]) for i, child in enumerate(self.children))
        return out


class HigherGRULayer(RecurrentLayer):
    """
        Implements the simple GRU layer.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 weight_initializer=None,
                 bias_initializer=None,
                 learn_init_state=False,
                 use_external_inps=True,
                 activ=None,
                 seq_len=None,
                 name=None):

        if name is None:
            raise ValueError("name should not be empty!")

        super(HigherGRULayer, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        if activ is None:
            self.activ = TT.tanh
        else:
            self.activ = activ
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.use_external_inps = use_external_inps
        self.learn_init_state = learn_init_state
        self.seq_len = seq_len
        self.init_params()

    def init_params(self):
        self.state_gater_before_proj = AffineLayer(n_in=self.n_out,
                                                   n_out=self.n_out,
                                                   weight_initializer=self.weight_initializer,
                                                   bias_initializer=self.bias_initializer,
                                                   name=self.pname("statebf_gater"))

        self.state_reset_before_proj = AffineLayer(n_in=self.n_out,
                                                   n_out=self.n_out,
                                                   weight_initializer=self.weight_initializer,
                                                   bias_initializer=self.bias_initializer,
                                                   name=self.pname("statebf_reset"))

        self.state_str_before_proj = AffineLayer(n_in=self.n_out,
                                                 n_out=self.n_out,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 name=self.pname("state_reset_ht"))

        cnames = ["reset_below",
                  "gater_below",
                  "state_below"]

        if not self.use_external_inps:
            nfout = len(cnames)
            self.cnames = map(lambda x: self.pname(x), cnames)
            self.controller_inps = ForkLayer(n_in=self.n_in,
                                             n_outs=[self.n_out for _ in xrange(nfout)],
                                             weight_initializer=self.weight_initializer,
                                             bias_initializer=self.bias_initializer,
                                             names=self.cnames)

        self.children = [self.state_gater_before_proj,
                         self.state_reset_before_proj,
                         self.state_str_before_proj]

        if not self.use_external_inps:
            self.children += [self.controller_inps]

        self.merge_params()
        self.str_params()

    def __step(self,
              state_before=None,
              state_fbefore=None,
              reset_below=None,
              gater_below=None,
              state_below=None,
              maskq=None,
              maskf=None,
              **kwargs):

        if maskq and maskf:
            ndim_diff = state_before.ndim - maskq.ndim

            if state_before.ndim == 2 and ndim_diff == 1:
                maskq = maskq.dimshuffle(0, 'x')
                maskf = maskf.dimshuffle(0, 'x')
            elif state_before.ndim == 3 and ndim_diff == 1:
                maskq = maskq.dimshuffle(0, 1, 'x')
                maskf = maskf.dimshuffle(0, 1, 'x')

            state_before = (1 - maskf) * state_before + maskf * state_fbefore

        state_reset = self.state_reset_before_proj.fprop(state_before)
        state_gater = self.state_gater_before_proj.fprop(state_before)
        reset = Sigmoid(reset_below + state_reset)
        state_state = self.state_str_before_proj.fprop(reset * state_before)
        gater = Sigmoid(gater_below + state_gater)

        h = self.activ(state_state + state_below)
        h_t = (1 - gater) * state_before + gater * h

        if maskq and maskf:
            hf_t = (1 - maskf) * state_fbefore + h_t * maskf
            h_t = maskq * h_t + (1 - maskq) * hf_t

        return h_t, hf_t

    def fprop(self, inps, maskq=None, maskf=None, batch_size=None):
        use_mask = 0 if (maskq is None or maskf is None) else 1
        if batch_size is not None:
            self.batch_size = batch_size

        if not self.use_external_inps:
            outs = self.controller_inps.fprop(inps)
            reset_below = outs[self.cnames[0]]
            gater_below = outs[self.cnames[1]]
            state_below = outs[self.cnames[2]]
        else:
            assert isinstance(inps, list)
            reset_below = inps[0]
            gater_below = inps[1]
            state_below = inps[2]

        def step_callback(*args):
            def lst_to_dict(lst):
                return {p.name: p for p in lst}

            reset_below, gater_below, state_below = args[0], args[1], args[2]

            if use_mask:
                mq = args[3]
                mf = args[4]
                step_res = self.__step(reset_below=reset_below,
                                       gater_below=gater_below,
                                       state_below=state_below,
                                       maskq=mq,
                                       maskf=mf,
                                       state_before=args[5],
                                       state_fbefore=args[6],
                                       **lst_to_dict(args[7:]))
            else:
                step_res = self.__step(reset_below=reset_below,
                                       gater_below=gater_below,
                                       state_below=state_below,
                                       state_before=args[3],
                                       **lst_to_dict(args[4:]))
            return step_res

        seqs = [reset_below, gater_below, state_below]
        if maskq is not None:
            if maskq.ndim == 3:
                maskq = maskq.reshape((maskq.shape[0], -1))
            seqs += [maskq]

        if maskf is not None:
            if maskf.ndim == 3:
                maskf = maskf.reshape((maskf.shape[0], -1))

            seqs += [maskf]

        if inps[0].ndim == 3 or (inps[0].ndim == 2 and \
                "int" in inps[0].dtype) or (maskf.ndim == 2):

            seqs[:-2] = map(lambda x: x.reshape((maskq.shape[0],
                                                 maskq.shape[1],
                                                 -1)), seqs[:-2])

            h0 = TT.alloc(as_floatX(0), maskq.shape[1],
                                        self.n_out)
            h0f = TT.alloc(as_floatX(0), maskq.shape[1],
                                        self.n_out)
        else:
            seqs = map(lambda x: x.reshape((maskq.shape[0],
                                            -1)), seqs)
            if self.batch_size == 1:
                h0 = TT.alloc(as_floatX(0), self.n_out)

                h0f = TT.alloc(as_floatX(0), self.n_out)
            else:
                h0 = TT.alloc(as_floatX(0), maskq.shape[1],
                                            self.n_out)
                h0f = TT.alloc(as_floatX(0), maskq.shape[1],
                                            self.n_out)

        if self.seq_len is None:
            n_steps = inps[0].shape[0]
        else:
            n_steps = self.seq_len

        non_sequences = self.params.values
        rval, updates = theano.scan(step_callback,
                                    sequences=seqs,
                                    outputs_info=[h0, h0f],
                                    n_steps=n_steps,
                                    non_sequences=non_sequences,
                                    strict=True)

        self.updates = updates
        return rval


class GRULayer(RecurrentLayer):
    """
       Implements the simple GRU layer.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 weight_initializer=None,
                 bias_initializer=None,
                 learn_init_state=False,
                 use_external_inps=True,
                 activ=None,
                 seq_len=None,
                 name=None):

        if name is None:
            raise ValueError("name should not be empty!")

        super(GRULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        if activ is None:
            self.activ = TT.tanh
        else:
            self.activ = activ

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.use_external_inps = use_external_inps
        self.learn_init_state = learn_init_state
        self.seq_len = seq_len
        self.init_params()

    def init_params(self):
        self.state_gater_before_proj = AffineLayer(n_in=self.n_out,
                                                   n_out=self.n_out,
                                                   weight_initializer=self.weight_initializer,
                                                   bias_initializer=self.bias_initializer,
                                                   name=self.pname("statebf_gater"))

        self.state_reset_before_proj = AffineLayer(n_in=self.n_out,
                                                   n_out=self.n_out,
                                                   weight_initializer=self.weight_initializer,
                                                   bias_initializer=self.bias_initializer,
                                                   name=self.pname("statebf_reset"))

        self.state_str_before_proj = AffineLayer(n_in=self.n_out,
                                                 n_out=self.n_out,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 name=self.pname("statebf_below"))
        cnames = ["reset_below",
                  "gater_below",
                  "state_below"]

        if not self.use_external_inps:
            nfout = len(cnames)
            self.cnames = map(lambda x: self.pname(x), cnames)
            self.controller_inps = ForkLayer(n_in=self.n_in,
                                             n_outs=[self.n_out for _ in xrange(nfout)],
                                             weight_initializer=self.weight_initializer,
                                             bias_initializer=self.bias_initializer,
                                             names=self.cnames)

        self.children = [self.state_gater_before_proj,
                         self.state_reset_before_proj,
                         self.state_str_before_proj]

        if not self.use_external_inps:
            self.children += [self.controller_inps]

        self.merge_params()
        self.str_params()

    def __step(self,
              state_before=None,
              reset_below=None,
              gater_below=None,
              state_below=None,
              mask=None,
              **kwargs):

        state_reset = self.state_reset_before_proj.fprop(state_before)
        state_gater = self.state_gater_before_proj.fprop(state_before)
        reset = Sigmoid(reset_below + state_reset)
        state_state = self.state_str_before_proj.fprop(reset * state_before)
        gater = Sigmoid(gater_below + state_gater)

        h = self.activ(state_state + state_below)
        h_t = (1 - gater) * state_before + gater * h

        if mask:
            ndim_diff = h_t.ndim - mask.ndim
            if h_t.ndim == 2 and ndim_diff == 1:
                mask = mask.dimshuffle(0, 'x')
            elif h_t.ndim == 3 and ndim_diff == 1:
                mask = mask.dimshuffle(0, 1, 'x')
            h_t = (1 - mask) * state_before + h_t  * mask
        return h_t

    def fprop(self, inps, mask=None, batch_size=None):
        use_mask = 0 if mask is None else 1
        if batch_size is not None:
            self.batch_size = batch_size

        if not self.use_external_inps:
            outs = self.controller_inps.fprop(inps)
            reset_below = outs[self.cnames[0]]
            gater_below = outs[self.cnames[1]]
            state_below = outs[self.cnames[2]]
        else:
            assert isinstance(inps, list)
            reset_below = inps[0]
            gater_below = inps[1]
            state_below = inps[2]

        def step_callback(*args):
            def lst_to_dict(lst):
                return {p.name: p for p in lst}

            reset_below, gater_below, state_below = args[0], args[1], args[2]
            if use_mask:
                m = args[3]
                step_res = self.__step(reset_below=reset_below,
                                       gater_below=gater_below,
                                       state_below=state_below,
                                       mask=m,
                                       state_before=args[4],
                                       **lst_to_dict(args[5:]))
            else:
                step_res = self.__step(reset_below=reset_below,
                                       gater_below=gater_below,
                                       state_below=state_below,
                                       state_before=args[4],
                                       **lst_to_dict(args[5:]))
            return step_res

        seqs = [reset_below, gater_below, state_below]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.reshape((mask.shape[0], -1))
            mask = mask.dimshuffle(0, 1, 'x')
            seqs += [mask]

        if inps[0].ndim == 3 or (inps[0].ndim == 2 and "int" in inps[0].dtype) or (mask.ndim == 2):
            seqs[:-1] = map(lambda x: x.reshape((mask.shape[0],
                                                 mask.shape[1],
                                                 -1)), seqs[:-1])
            h0 = TT.alloc(as_floatX(0), mask.shape[1],
                                        self.n_out)
        else:
            seqs = map(lambda x: x.reshape((mask.shape[0],
                                            -1)), seqs)
            if self.batch_size == 1:
                h0 = TT.alloc(as_floatX(0), self.n_out)
            else:
                h0 = TT.alloc(as_floatX(0), mask.shape[1],
                                            self.n_out)

        if self.seq_len is None:
            n_steps = inps[0].shape[0]
        else:
            n_steps = self.seq_len

        non_sequences = self.params.values
        rval, updates = theano.scan(step_callback,
                                    sequences=seqs,
                                    outputs_info=[h0],
                                    n_steps=n_steps,
                                    non_sequences=non_sequences,
                                    strict=True)
        self.updates = updates
        return rval


class RNNLayer(RecurrentLayer):
    """
        Implements the simple GRU layer.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 weight_initializer=None,
                 bias_initializer=None,
                 learn_init_state=False,
                 use_external_inps=True,
                 activ=None,
                 seq_len=None,
                 name=None):

        if name is None:
            raise ValueError("name should not be empty!")

        super(RNNLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        if activ is None:
            self.activ = Rect
        else:
            self.activ = activ

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.use_external_inps = use_external_inps
        self.learn_init_state = learn_init_state
        self.seq_len = seq_len
        self.init_params()

    def init_params(self):

        self.state_str_before_proj = AffineLayer(n_in=self.n_out,
                                                 n_out=self.n_out,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 name=self.pname("state_before_ht"))

        self.children = [ self.state_str_before_proj ]

        if not self.use_external_inps:
            self.controller_inps = AffineLayer(n_in=self.n_in,
                                               n_out=self.n_out,
                                               use_bias=False,
                                               weight_initializer=self.weight_initializer,
                                               bias_initializer=self.bias_initializer,
                                               name=self.pname("state_below_ht"))
            self.children += [ self.controller_inps ]

        self.merge_params()
        self.str_params()

    def __step(self,
              state_before=None,
              state_below=None,
              mask=None,
              **kwargs):

        state_state = self.state_str_before_proj.fprop(state_before)
        h = self.activ(state_state + state_below.reshape(state_state.shape))

        if mask:
            ndim_diff = h.ndim - mask.ndim
            if h.ndim == 2 and ndim_diff == 1:
                mask = mask.dimshuffle(0, 'x')
            elif h.ndim == 3 and ndim_diff == 1:
                mask = mask.dimshuffle(0, 1, 'x')
            h = (1 - mask) * state_before + h  * mask

        return h

    def fprop(self, inps, mask=None, batch_size=None):
        use_mask = 0 if mask is None else 1

        if batch_size is not None:
            self.batch_size = batch_size

        if not self.use_external_inps:
            out = self.controller_inps.fprop(inps)
            state_below = out
        else:
            state_below = inps

        def step_callback(*args):
            def lst_to_dict(lst):
                return {p.name: p for p in lst}

            state_below = args[0]
            if use_mask:
                m = args[1]
                step_res = self.__step(state_below=state_below,
                                       mask=m,
                                       state_before=args[2],
                                       **lst_to_dict(args[3:]))
            else:
                step_res = self.__step(state_below=state_below,
                                       state_before=args[1],
                                       **lst_to_dict(args[2:]))
            return step_res

        seqs = [ state_below ]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.reshape((mask.shape[0], -1))

            mask = mask.dimshuffle(0, 1, 'x')
            seqs += [ mask ]

        if inps[0].ndim == 3 or (inps[0].ndim == 2 and "int" in inps[0].dtype) or (mask.ndim == 2):
            seqs[:-1] = map(lambda x: x.reshape((mask.shape[0],
                                                 mask.shape[1],
                                                 -1)), seqs[:-1])

            h0 = TT.alloc(as_floatX(0),
                          mask.shape[1],
                          self.n_out)
        else:
            seqs = map(lambda x: x.reshape((mask.shape[0],
                                            -1)), seqs)
            if self.batch_size == 1:
                h0 = TT.alloc(as_floatX(0), self.n_out)
            else:
                h0 = TT.alloc(as_floatX(0), mask.shape[1],
                                            self.n_out)

        if self.seq_len is None:
            n_steps = inps[0].shape[0]
        else:
            n_steps = self.seq_len

        non_sequences = self.params.values
        rval, updates = theano.scan(step_callback,
                                    sequences=seqs,
                                    outputs_info=[h0],
                                    n_steps=n_steps,
                                    non_sequences=non_sequences,
                                    strict=True)
        self.updates = updates
        return rval


class ConvMLPLayer(Layer):
    """
        Convolutional MLP Layer.
    """
    def __init__(self,
                 patch_size=8*8,
                 npatches=64,
                 n_hids=100,
                 n_out=100,
                 activ=None,
                 binit_vals=None,
                 out_activ=None,
                 use_deepmind_pooling=False,
                 weight_initializer=None,
                 bias_initializer=None,
                 name=None):

        if name is None:
            raise ValueError("name property should not be empty!")

        super(ConvMLPLayer, self).__init__()

        self.patch_size = patch_size
        self.npatches = npatches
        self.use_deepmind_pooling = use_deepmind_pooling

        if activ is None:
            self.activ = Rect
        else:
            self.activ = activ

        if out_activ is None:
            self.out_activ = Rect
        else:
            self.out_activ = out_activ

        self.n_out = n_out
        self.n_hids = n_hids
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.binit_vals = binit_vals
        self.name = name
        self.init_params()

    def init_params(self):
        self.patch_in = AffineLayer(n_in=self.patch_size,
                                    n_out=self.n_hids,
                                    weight_initializer=self.weight_initializer,
                                    bias_initializer=self.bias_initializer,
                                    init_bias_val=self.binit_vals,
                                    name=self.pname("patch_in"))

        self.state_out = AffineLayer(n_in=self.n_hids,
                                     n_out=self.n_out,
                                     weight_initializer=self.weight_initializer,
                                     bias_initializer=self.bias_initializer,
                                     init_bias_val=self.binit_vals,
                                     name=self.pname("patch_out"))

        self.children = [self.patch_in, self.state_out]
        self.merge_params()
        self.str_params()

    def fprop(self, inp):
        if inp.ndim != 3:
            new_inp = inp.reshape((-1, self.npatches, self.patch_size))
        else:
            new_inp = inp

        patch_hid = self.activ(self.patch_in.fprop(new_inp))
        patch_out = self.out_activ(self.state_out.fprop(patch_hid))

        if self.use_deepmind_pooling:
            logger.info("Using deep mind's global pooling.")
            patch_out = patch_out.reshape((-1, self.npatches, self.n_out))
            patch_out = patch_out.max(-1).reshape((-1, self.npatches))
        return patch_out


class LSTMLayer(RecurrentLayer):
    """
    Implements the simple GRU layer.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 weight_initializer=None,
                 bias_initializer=None,
                 learn_init_state=False,
                 use_external_inps=True,
                 activ=None,
                 seq_len=None,
                 name=None):

        if name is None:
            raise ValueError("name should not be empty!")

        super(LSTMLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if activ is None:
            self.activ = TT.tanh
        else:
            self.activ = activ
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.use_external_inps = use_external_inps
        self.learn_init_state = learn_init_state
        self.seq_len = seq_len
        self.init_params()

    def init_params(self):
        st_bf_names = ["forget_stbf",
                       "input_stbf",
                       "out_stbf",
                       "cell_stbf"]

        self.sbf_names = map(lambda x: self.pname(x), st_bf_names)
        binit_vals = [-1e-5 for i in xrange(len(st_bf_names))]
        nouts = [self.n_out for i in xrange(len(st_bf_names))]

        self.state_before_fork_layer = ForkLayer(n_in=self.n_out,
                                                 n_outs=nouts,
                                                 weight_initializer=self.weight_initializer,
                                                 bias_initializer=self.bias_initializer,
                                                 init_bias_vals=binit_vals,
                                                 names=self.sbf_names)
        self.children = [self.state_before_fork_layer]
        self.merge_params()
        self.str_params()

    def __step(self,
               state_before=None,
               cell_before=None,
               forget_below=None,
               input_below=None,
               output_below=None,
               state_below=None,
               mask=None,
               **kwargs):

        state_fork_outs = self.state_before_fork_layer.fprop(state_before)
        inp = Sigmoid(input_below + state_fork_outs[self.sbf_names[1]])
        output = Sigmoid(output_below + state_fork_outs[self.sbf_names[2]])
        forget = Sigmoid(forget_below + state_fork_outs[self.sbf_names[0]])
        cell = Tanh(state_below + state_fork_outs[self.sbf_names[3]])

        c_t = inp * cell + forget * cell_before
        h_t = output * self.activ(c_t)

        if mask:
            ndim_diff = h_t.ndim - mask.ndim
            if h_t.ndim == 2 and ndim_diff == 1:
                mask = mask.dimshuffle(0, 'x')
            elif h_t.ndim == 3 and ndim_diff == 1:
                mask = mask.dimshuffle(0, 1, 'x')
            h_t = (1 - mask) * state_before + h_t  * mask
            c_t = (1 - mask) * cell_before + c_t * mask

        return h_t, c_t

    def fprop(self,
              inps,
              mask=None,
              batch_size=None):

        use_mask = 0 if mask is None else 1
        if batch_size is not None:
            self.batch_size = batch_size

        if not self.use_external_inps:
            if self.controller_inps is None:
                raise RuntimeError("Controller inputs should not be None.")

            outs = self.controller_inps.fprop(inps)
            forget_below = outs[self.cnames[0]]
            input_below = outs[self.cnames[1]]
            output_below = outs[self.cnames[2]]
            cell_below = outs[self.cnames[3]]
        else:
            assert isinstance(inps, list)
            forget_below = inps[0]
            input_below = inps[1]
            output_below = inps[2]
            cell_below = inps[3]

        def step_callback(*args):
            def lst_to_dict(lst):
                return {p.name: p for p in lst}
            forget_below, input_below, output_below, cell_below = args[0], \
                    args[1], args[2], args[3]

            if use_mask:
                m = args[4]
                spdict = lst_to_dict(args[7:])
                step_res = self.__step(forget_below=forget_below,
                                       input_below=input_below,
                                       output_below=output_below,
                                       state_below=cell_below,
                                       mask=m,
                                       state_before=args[5],
                                       cell_before=args[6],
                                       **spdict)
            else:
                spdict = lst_to_dict(args[6:])
                step_res = self.__step(forget_below=forget_below,
                                       input_below=input_below,
                                       output_below=output_below,
                                       state_below=cell_below,
                                       state_before=args[4],
                                       cell_before=args[5],
                                       **spdict)
            return step_res

        seqs = [forget_below, input_below, output_below, cell_below]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.reshape((mask.shape[0], -1))
            mask = mask.dimshuffle(0, 1, 'x')
            seqs += [mask]

        if inps[0].ndim == 3 or (inps[0].ndim == 2 and "int" in inps[0].dtype) or \
                (mask.ndim == 2):
            seqs[:-1] = map(lambda x: x.reshape((mask.shape[0],
                                                 mask.shape[1],
                                                 -1)), seqs[:-1])

            h0 = TT.alloc(as_floatX(0), mask.shape[1],
                                        self.n_out)
            c0 = TT.alloc(as_floatX(0), mask.shape[1],
                                        self.n_out)

        else:
            seqs = map(lambda x: x.reshape((mask.shape[0],
                                            -1)), seqs)
            if self.batch_size == 1:
                h0 = TT.alloc(as_floatX(0), self.n_out)
                c0 = TT.alloc(as_floatX(0), self.n_out)
            else:
                h0 = TT.alloc(as_floatX(0), mask.shape[1],
                                            self.n_out)
                c0 = TT.alloc(as_floatX(0), mask.shape[1],
                                            self.n_out)


        if self.seq_len is None:
            n_steps = inps[0].shape[0]
        else:
            n_steps = self.seq_len

        non_sequences = self.params.values
        rval, updates = theano.scan(step_callback,
                                    sequences=seqs,
                                    outputs_info=[h0, c0],
                                    n_steps=n_steps,
                                    non_sequences=non_sequences,
                                    strict=True)

        self.updates = updates
        return rval
