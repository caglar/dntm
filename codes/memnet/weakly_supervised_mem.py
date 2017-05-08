import theano
import theano.tensor as TT
from core.layers import (Layer,
                                BOWLayer,
                                AffineLayer)

from core.timer import Timer
from core.commons import Softmax, global_rng
from core.utils import safe_grad, as_floatX
from core.nnet_utils import nll
import logging

logger = logging.getLogger(__name__)
logger.disabled = False


class WeaklySupervisedMemoryNet(Layer):
    """
    An implementation of weakly supervised memory network paper.
    """
    def __init__(self,
                 n_in,
                 n_out,
                 bow_size,
                 weight_initializer=None,
                 use_index_jittering=False,
                 bias_initializer=None,
                 max_fact_len=12,
                 max_seq_len=250,
                 dropout=None,
                 batch_size=None,
                 learning_rule=None,
                 share_inp_out_weights=False,
                 n_steps=1,
                 inps=None,
                 use_noise=False,
                 theano_function_mode=None,
                 rng=None,
                 name=None):

        self.n_in = n_in
        self.n_out = n_out
        self.bow_size = bow_size
        self.use_index_jittering = use_index_jittering
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.share_inp_out_weights = share_inp_out_weights
        self.rng = rng
        self.inps = inps
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rule = learning_rule
        self.theano_function_mode = theano_function_mode
        self.eps = 1e-7
        self.max_fact_len = max_fact_len
        self.max_seq_len = max_seq_len
        self.n_steps = n_steps
        self.use_noise = use_noise
        self.name = name
        assert n_steps > 0, "Illegal value has been provided for n_steps."
        self.train_timer = Timer("Training function")
        self.grads_timer = Timer("Computing the grads")
        self.updates = {}

    def init_params(self, use_noise=False, mdl_name=None):
        if not hasattr(self, "children") or not self.children:
            self.children = []
            self.inp_bow_layer = BOWLayer(n_in=self.n_in,
                                        n_out=self.bow_size,
                                        seq_len=self.max_fact_len,
                                        use_inv_cost_mask=False,
                                        weight_initializer=self.weight_initializer,
                                        bias_initializer=self.bias_initializer,
                                        use_average=False,
                                        name=self.pname("bow_layer"))
            self.inp_bow_layers = [self.inp_bow_layer]

            self.out_bow_layer = BOWLayer(n_in=self.n_in,
                                        n_out=self.bow_size,
                                        seq_len=self.max_fact_len,
                                        use_inv_cost_mask=False,
                                        weight_initializer=self.weight_initializer,
                                        bias_initializer=self.bias_initializer,
                                        use_average=False,
                                        name=self.pname("out_bow_layer"))

            self.out_bow_layers = [self.out_bow_layer]

            if not self.share_inp_out_weights:
                for i in xrange(1, self.n_steps):
                    self.inp_bow_layers += [BOWLayer(n_in=self.n_in,
                                            n_out=self.bow_size,
                                            seq_len=self.max_fact_len,
                                            use_inv_cost_mask=False,
                                            weight_initializer=self.weight_initializer,
                                            bias_initializer=self.bias_initializer,
                                            use_average=False,
                                            name=self.pname("bow_layer_" + str(i)))]

                    self.out_bow_layers += [BOWLayer(n_in=self.n_in,
                                                    n_out=self.bow_size,
                                                    use_inv_cost_mask=False,
                                                    seq_len=self.max_fact_len,
                                                    weight_initializer=self.weight_initializer,
                                                    bias_initializer=self.bias_initializer,
                                                    use_average=False,
                                                    name=self.pname("out_bow_layer_" + str(i)))]

            self.q_embed = BOWLayer(n_in=self.n_in,
                                    n_out=self.bow_size,
                                    use_inv_cost_mask=False,
                                    seq_len=self.max_fact_len,
                                    weight_initializer=self.weight_initializer,
                                    bias_initializer=self.bias_initializer,
                                    use_average=False,
                                    name=self.pname("q_embed"))

            self.out_layer = AffineLayer(n_in=self.bow_size,
                                        n_out=self.n_out,
                                        weight_initializer=self.weight_initializer,
                                        bias_initializer=self.bias_initializer,
                                        name=self.pname("out_layer"))

            self.children.extend(self.inp_bow_layers)
            self.children.extend(self.out_bow_layers)
            self.children.append(self.out_layer)
            self.children.append(self.q_embed)
            self.merge_params()

            # These are the parameters for the temporal encoding thing:
            self.T_ins = []
            self.T_outs = []

            nsteps = 1 if self.share_inp_out_weights else self.n_steps
            #"""
            for i in xrange(nsteps):
                T_in = self.weight_initializer(self.max_seq_len, self.bow_size)
                self.params[self.pname("TE_in_%d" % i)] = T_in
                self.T_ins.append(self.params[self.pname("TE_in_%d" % i)])
                T_out = self.weight_initializer(self.max_seq_len, self.bow_size)
                self.params[self.pname("TE_out_%d" % i)] = T_out
                self.T_outs.append(self.params[self.pname("TE_out_%d" % i)])
            #"""
        if mdl_name:
            logger.info("Reloading model from %s." % mdl_name)
            self.params.load(mdl_name)
            [child.use_params(self.params) for child in self.children]

    def get_cost(self, use_noise=False, mdl_name=None):
        X = self.inps[0]
        q = self.inps[1]
        y = self.inps[2]
        mask = self.inps[3]
        cmask = None
        probs = self.fprop(X, q, cmask=cmask,
                              mask=mask, use_noise=use_noise,
                              mdl_name=mdl_name)
        self.cost, self.errors = nll(y, probs)
        return self.cost, self.errors

    def get_inspect_fn(self, mdl_name=None):
        logger.info("Compiling inspect function.")
        probs, ntm_outs = self.fprop(use_noise=False, mdl_name=mdl_name)

        inspect_fn = theano.function([self.inps[0],
                                      self.inps[1],
                                      self.inps[2],
                                      self.inps[3]],
                                      ntm_outs + [probs],
                                      on_unused_input='ignore',
                                      name=self.pname("inspect_fn"))

        return inspect_fn

    def get_valid_fn(self, mdl_name=None):
        logger.info("Compiling validation function.")
        self.cost, self.errors = self.get_cost(use_noise=False,
                                               mdl_name=mdl_name)
        valid_fn = theano.function(self.inps,
                                   [self.cost, self.errors],
                                   on_unused_input='ignore',
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

        cost, errors = self.get_cost(use_noise=self.use_noise,
                                     mdl_name=mdl_name)

        params = self.params.values
        logger.info("Computing the gradients.")
        self.grads_timer.start()
        grads = safe_grad(cost, params)
        gnorm = sum(grad.norm(2) for _, grad in grads.iteritems())

        updates, norm_up, param_norm = self.learning_rule.get_updates(learning_rate=lr,
                                                                      grads=grads)
        self.grads_timer.stop()
        logger.info(self.grads_timer)

        if not self.updates:
            self.updates = self.updates.update(updates)

        logger.info("Compiling the training function.")
        self.train_timer.start()
        self.updates = updates
        outs = [self.cost, gnorm, norm_up, param_norm]
        outs += [self.errors]

        train_fn = theano.function(self.inps,
                                   outs,
                                   updates=updates,
                                   mode=self.theano_function_mode,
                                   on_unused_input='ignore',
                                   name=self.pname("train_fn"))

        self.train_timer.stop()
        logger.info(self.train_timer)
        return train_fn

    def __get_bow_inps(self, x, q,
                       mask=None,
                       use_noise=False):
        inp_bow_outs, out_bow_outs = [], []
        nsteps = 1 if self.share_inp_out_weights else self.n_steps
        for i in xrange(nsteps):
            inp_bow_outs.append(self.inp_bow_layers[i].fprop(x, amask=mask,
                                deterministic=not use_noise))
            out_bow_outs.append(self.out_bow_layers[i].fprop(x, amask=mask,
                                deterministic=not use_noise))
        return inp_bow_outs, out_bow_outs

    def dot_componentwise(self, x, u_t):
        if x.ndim == 3:
            u_t = u_t.dimshuffle('x', 0, 1)
        res = (x * u_t).sum(-1)
        return res

    def fprop(self, x, q,
              mask=None,
              qmask=None,
              cmask=None,
              use_noise=False,
              mdl_name=None):

        self.init_params(use_noise=use_noise, mdl_name=mdl_name)
        q_emb = self.q_embed.fprop(q, deterministic=not use_noise)
        amask = None

        if mask is not None and cmask is not None:
            amask = mask * TT.eq(cmask, 0)

        inp_bow_outs, out_bow_outs = self.__get_bow_inps(x, q,
                                                         mask=amask,
                                                         use_noise=use_noise)
        u_t = q_emb
        v_t = None
        if mask.ndim == 2 and \
                inp_bow_outs[0].ndim == 3:
            mask = mask.dimshuffle(0, 1, 'x')

        for i in xrange(self.n_steps):
            if not self.share_inp_out_weights:
                inp_bow = mask * (inp_bow_outs[i] + self.T_ins[i].dimshuffle(0, 'x', 1))
                out_bow = mask * (out_bow_outs[i] + self.T_outs[i].dimshuffle(0, 'x', 1))
            else:
                inp_bow = mask * (inp_bow_outs[0] + self.T_ins[0].dimshuffle(0, 'x', 1))
                out_bow = mask * (out_bow_outs[0] + self.T_outs[0].dimshuffle(0, 'x', 1))

            if u_t.ndim == 2:
                u_t = u_t.dimshuffle(0, 1, 'x')

            sims = self.dot_componentwise(inp_bow, u_t)
            pre_soft = mask.dimshuffle(0, 1) * TT.exp(sims - sims.max(0))
            ps = pre_soft / pre_soft.sum(axis=0, keepdims=True)
            ps = ps.dimshuffle(0, 1, 'x')
            v_t = (out_bow * ps).sum(0)
            u_t = u_t.dimshuffle(0, 1) + v_t

        new_out = u_t
        pre_logit = self.out_layer.fprop(new_out)
        probs = Softmax(pre_logit)
        return probs
