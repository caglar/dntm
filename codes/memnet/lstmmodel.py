from collections import OrderedDict
import logging

import theano
import theano.tensor as TT

from core.layers import (AffineLayer,
                                ForkLayer,
                                GRULayer,
                                LSTMLayer,
                                BOWLayer)

from core.parameters import Parameters
from core.basic import Model
from core.utils import safe_grad, as_floatX
from core.commons import Leaky_Rect, Rect, Softmax
from core.timer import Timer
from core.operators import Dropout

logger = logging.getLogger(__name__)
logger.disabled = False


class LSTMModel(Model):

    def __init__(self,
                 n_in,
                 n_hids,
                 bow_size,
                 n_out,
                 inps=None,
                 dropout=None,
                 seq_len=None,
                 learning_rule=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 learn_h0=False,
                 deepout=None,
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
        self.bow_size = bow_size
        self.inps = inps
        self.noise = noise
        self.seq_len = seq_len
        self.dropout = dropout
        self.use_cost_mask = use_cost_mask
        self.learning_rule = learning_rule
        self.bias_initializer = bias_initializer
        self.learn_h0 = learn_h0
        self.use_average = use_average
        self.deepout = deepout
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
        if bow_size is None:
            raise ValueError("bow_size should be specified.")

        if name is None:
            raise ValueError("name should not be empty.")

        self.reset()

        self.name = name

    def reset(self):
        self.children = []
        self.params = Parameters()
        self.bow_layer = None
        self.lstm_layer = None
        self.out_layer = None
        self.bowup_layer = None
        self.hint_layer = None
        self.updates = OrderedDict({})

    def build_model(self, use_noise=False, mdl_name=None):
        if not self.bow_layer:
            self.bow_layer = BOWLayer(n_in=self.n_in,
                                      n_out=self.bow_size,
                                      seq_len=12,
                                      weight_initializer=self.weight_initializer,
                                      bias_initializer=self.bias_initializer,
                                      use_average=False,
                                      name=self.pname("bow_layer"))

        if self.deepout:
            self.deepout_layer_qbow = AffineLayer(n_in=self.bow_size,
                                                  n_out=self.deepout,
                                                  weight_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  name=self.pname("deepout_qbow"))

            self.deepout_layer_ht = AffineLayer(n_in=self.n_hids,
                                                n_out=self.deepout,
                                                weight_initializer=self.weight_initializer,
                                                bias_initializer=self.bias_initializer,
                                                name=self.pname("deepout_ht"))
            self.out_layer_in = self.deepout

        if not self.bowup_layer:
            cnames = ["forget_below", "input_below", "out_below", "cell_below"]

            nfout = len(cnames)
            self.cnames = map(lambda x: self.pname(x), cnames)

            self.bowup_layer = ForkLayer(n_in=self.bow_size,
                                         n_outs=tuple([self.n_hids for i in xrange(nfout)]),
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         names=self.cnames)


        if not self.lstm_layer:
            self.lstm_layer = LSTMLayer(n_in=self.n_hids,
                                        n_out=self.n_hids,
                                        seq_len=self.seq_len,
                                        weight_initializer=self.weight_initializer,
                                        bias_initializer=self.bias_initializer,
                                        activ=self.activ,
                                        learn_init_state=self.learn_h0,
                                        name=self.pname("lstm_layer"))

        if not self.out_layer:
            self.out_layer = AffineLayer(n_in=self.out_layer_in,
                                         n_out=self.n_out,
                                         noise=self.noise,
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         name=self.pname("ntm_out"))

        if not self.children:
            self.children.append(self.bowup_layer)
            self.children.append(self.bow_layer)
            self.children.append(self.lstm_layer)
            self.children.append(self.out_layer)

            if self.deepout:
                self.children.append(self.deepout_layer_qbow)
                self.children.append(self.deepout_layer_ht)

            self.merge_params()

        if mdl_name:
            logger.info("Reloading the model from %s. " % mdl_name)
            self.params.load(mdl_name)
            [child.use_params(self.params) for child in self.children]

    def get_cost(self, use_noise=False, mdl_name=None):
        probs, _ = self.fprop(use_noise=use_noise, mdl_name=mdl_name)
        y = self.inps[1]
        cmask = None

        if self.use_cost_mask :
            cmask = self.inps[3]

        self.cost, self.errors = nll(y, probs, cost_mask=cmask)
        return self.cost, self.errors

    def get_train_fn(self, lr=None, mdl_name=None):
        if lr is None:
            lr = self.eps

        cost, errors = self.get_cost(use_noise=self.use_noise,
                                     mdl_name=mdl_name)

        params = self.params.values
        logger.info("Computing the gradient graph.")
        self.grads_timer.start()
        grads = safe_grad(cost, params)
        gnorm = sum(grad.norm(2) for _, grad in grads.iteritems())

        updates, norm_up, param_norm = self.learning_rule.get_updates(learning_rate=lr,
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

        if use_cmask:
            cmask = inps[3]
            qmask = inps[4]

        assert (3 + sum([use_mask, use_cmask])) == len(inps), "inputs have illegal shape."

        if cmask is not None:
            m = mask * TT.eq(cmask.reshape((cmask.shape[0], cmask.shape[1])), 0)
        else:
            raise ValueError("Mask for the answers should not be empty.")

        bow_out = self.bow_layer.fprop(X, amask = m, qmask=qmask, deterministic=not use_noise)
        new_bow = TT.roll(bow_out, 1, axis=0)
        new_bow = TT.set_subtensor(new_bow[0], as_floatX(0))
        bow_outs = self.bowup_layer.fprop(bow_out, deterministic=not use_noise)

        forget_below = bow_outs[self.cnames[0]].reshape((X.shape[1], X.shape[2], -1))
        input_below = bow_outs[self.cnames[1]].reshape((X.shape[1], X.shape[2], -1))
        output_below = bow_outs[self.cnames[2]].reshape((X.shape[1], X.shape[2], -1))
        cell_below = bow_outs[self.cnames[3]].reshape((X.shape[1], X.shape[2], -1))

        inps = [forget_below, input_below, output_below, cell_below]

        h, c = self.lstm_layer.fprop(inps=inps, mask=mask, batch_size=self.batch_size)
        if self.deepout:
            h_deepout = self.deepout_layer_ht.fprop(h)
            emb_deepout = self.deepout_layer_qbow.fprop(new_bow)
            z = Leaky_Rect(h_deepout + emb_deepout, 0.01)
            if self.dropout:
                dropOp = Dropout(dropout_prob=self.dropout)
                z = dropOp(z, deterministic=not use_noise)
        else:
            z = h
            if self.dropout:
                dropOp = Dropout(dropout_prob=self.dropout)
                z = dropOp(z, deterministic=not use_noise)

        out_layer = self.out_layer.fprop(z, deterministic=not use_noise)
        self.probs = Softmax(out_layer)
        return self.probs, h
