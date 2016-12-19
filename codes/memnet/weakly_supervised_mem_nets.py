from collections import OrderedDict
import logging

import theano
import theano.tensor as TT

from core.layers import (AffineLayer,
                                ForkLayer,
                                BOWLayer)

from core.parameters import Parameters
from core.basic import Model
from core.nnet_utils import kl, nll
from core.utils import safe_grad, as_floatX
from core.commons import Leaky_Rect, Rect, Softmax
from core.timer import Timer
from core.operators import Dropout

logger = logging.getLogger(__name__)
logger.disabled = False

class WeaklySupervisedMemoryNetwork(Model):
    """
        This is a class for weakly
        supervised memory network.
    """
    def __init__(self,
                 n_in,
                 n_hids,
                 low_gru_size,
                 n_out,
                 inps=None,
                 n_layers=None,
                 dropout=None,
                 seq_len=None,
                 learning_rule=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 activ=None,
                 use_cost_mask=True,
                 noise=False,
                 use_hint_layer=False,
                 use_average=False,
                 theano_function_mode=None,
                 use_positional_encoding=False,
                 use_inv_cost_mask=False,
                 batch_size=32,
                 use_noise=False,
                 name=None):

        self.n_in = n_in
        self.n_hids = n_hids
        self.n_out = n_out
        self.low_gru_size = low_gru_size
        self.n_layers = n_layers
        self.inps = inps
        self.noise = noise
        self.seq_len = seq_len
        self.use_cost_mask = use_cost_mask
        selfearning_rule = learning_rule
        self.dropout = dropout
        self.use_average = use_average
        self.batch_size = batch_size
        self.use_noise = use_noise

        self.train_timer = Timer("Training function")
        self.grads_timer = Timer("Computing the grads")
        self.theano_function_mode = theano_function_mode

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.use_average = use_average
        self.use_positional_encoding = use_positional_encoding
        self.use_inv_cost_mask = use_inv_cost_mask
        self.eps = 1e-8

        self.activ = activ
        self.out_layer_in = self.n_hids

        if name is None:
            raise ValueError("name should not be empty.")

        self.reset()
        self.name = name

    def reset(self):
        self.children = []
        self.params = Parameters()
        self.grulow_layer = None
        self.low_gru_layer = None
        self.gruup_layer = None

        self.gru_layer = None
        self.out_layer = None

        self.hint_layer = None
        self.bow_input = None
        self.bow_output = None

        self.updates = OrderedDict({})

    def build_model(self,
                    use_noise=False,
                    mdl_name=None):

        self.bowin_layer = BOWLayer(n_in=self.n_in,
                                    n_out=self.emb_size,
                                    noise=self.noise,
                                    weight_initializer=self.wight_initializer,
                                    bias_initializer=self.bias_initializer,
                                    seq_len=self.seq_len,
                                    name=self.pname("bowin_layer"))

        self.bowout_layer = BOWLayer(n_in=self.n_in,
                                     n_out=self.emb_size,
                                     noise=self.noise,
                                     weight_initializer=self.wight_initializer,
                                     bias_initializer=self.bias_initializer,
                                     seq_len=self.seq_len,
                                     name=self.pname("bowout_layer"))

        self.qembed_layer = BOWLayer(n_in=self.n_in,
                                     n_out=self.emb_size,
                                     noise=self.noise,
                                     weight_initializer=self.wight_initializer,
                                     bias_initializer=self.bias_initializer,
                                     seq_len=self.seq_len,
                                     name=self.pname("qembed_layer"))


        if not self.out_layer:
            self.out_layer = AffineLayer(n_in=self.out_layer_in,
                                         n_out=self.n_out,
                                         noise=self.noise,
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         name=self.pname("out_layer"))

        if not self.children:
            self.children.append(self.bowin_layer)
            self.children.append(self.bowout_layer)
            self.children.append(self.qembed_layer)
            self.children.append(self.out_layer)
            self.merge_params()

        if mdl_name:
            logger.info("Reloading the model from %s. " % mdl_name)
            self.params.load(mdl_name)
            [child.use_params(self.params) for child in self.children]

    def get_cost(self,
                 use_noise=False,
                 mdl_name=None):

        probs, _ = self.fprop(use_noise=use_noise, mdl_name=mdl_name)
        y = self.inps[1]
        cmask = None

        if self.use_cost_mask:
            cmask = self.inps[3]

        self.cost, self.errors = nll(y, probs, cost_mask=cmask)
        return self.cost, self.errors

    def get_train_fn(self,
                    lr=None,
                    mdl_name=None):

        if lr is None:
            lr = self.eps

        cost, errors = self.get_cost(use_noise=self.use_noise,
                                     mdl_name=mdl_name)

        params = self.params.values
        logger.info("Computing the gradient graph.")
        self.grads_timer.start()
        grads = safe_grad(cost, params)
        gnorm = sum(grad.norm(2) for _, grad in grads.iteritems())

        updates, norm_up, param_norm = \
                self.learning_rule.get_updates(learning_rate=lr,
                                               grads = grads)

        self.grads_timer.stop()
        logger.info(self.grads_timer)

        if not self.updates:
            self.updates = self.updates.update(updates)

        outs = [self.cost, gnorm, norm_up, param_norm]
        outs += [self.errors]

        train_fn = theano.function(self.inps,
                                   outs,
                                   updates=updates,
                                   mode=self.theano_function_mode,
                                   name=self.pname("train_fn"))

        self.train_timer.stop()
        logger.info(self.train_timer)
        return train_fn

    def get_inspect_fn(self, mdl_name=None):
        logger.info("Compiling inspect function.")
        probs, h = self.fprop(use_noise=False, mdl_name=mdl_name)
        inspect_fn = theano.function([self.inps[0], self.inps[2]],
                                     [h, probs],
                                     name=self.pname("inspect_fn"))

        return inspect_fn

    def get_valid_fn(self, mdl_name=None):
        logger.info("Compiling validation function.")

        self.cost, self.errors = self.get_cost(use_noise=False, mdl_name=mdl_name)

        valid_fn = theano.function(self.inps,
                                   [self.cost, self.errors],
                                   name=self.pname("valid_fn"))

        return valid_fn

    def fprop(self, inps=None,
              use_mask=True,
              use_cmask=True,
              use_noise=False,
              mdl_name=None):

        self.build_model(use_noise=use_noise, mdl_name=mdl_name)

        if not inps:
            inps = self.inps

        X = inps[0]

        if use_mask:
            mask = inps[2]
            qmask = inps[3]

        if use_cmask:
            cmask = inps[4]

        assert (3 + sum([use_mask, use_cmask])) == len(inps), "inputs have illegal shape."
        m0 = as_floatX(TT.gt(X, 0))

        if cmask is not None:
            m1 = mask * TT.eq(cmask, 0)
        else:
            raise ValueError("Mask for the answers should not be empty.")

        dropOp = None
        low_inp_shp = (X.shape[0], X.shape[1]*X.shape[2], -1)
        Xr = X.reshape(low_inp_shp)

        grulow_inps = self.grulow_layer.fprop(Xr, deterministic=not use_noise)

        linps = [low_reset_below, low_gater_below, low_state_below]
        inp_shp = (X.shape[1], X.shape[2], -1)

        h0 = self.low_gru_layer.fprop(inps=linps,
                                      mask=m0,
                                      batch_size=self.batch_size)

        h0 = m1.dimshuffle(0, 1, 'x') * (h0.reshape((X.shape[0],
                                                     X.shape[1],
                                                     X.shape[2],
                                                     -1))[-1]).reshape(inp_shp)

        if self.dropout:
            if dropOp is None:
                dropOp = Dropout(dropout_prob=self.dropout)
            h0 = dropOp(h0, deterministic=not use_noise)

        gruup_inps = self.gruup_layer.fprop(h0, deterministic=not use_noise)

        reset_below = gruup_inps.values()[0].reshape(inp_shp)
        gater_below = gruup_inps.values()[1].reshape(inp_shp)
        state_below = gruup_inps.values()[2].reshape(inp_shp)
        uinps = [reset_below, gater_below, state_below]

        h1, _ = self.gru_layer.fprop(inps=uinps,
                                     maskf=m1,
                                     maskq=qmask,
                                     batch_size=self.batch_size)

        if self.dropout:
            if dropOp is None:
                dropOp = Dropout(dropout_prob=self.dropout)
            h1 = dropOp(h1, deterministic=not use_noise)

        out_layer = self.out_layer.fprop(h1, deterministic=not use_noise)
        self.probs = Softmax(out_layer)
        return self.probs, h1

