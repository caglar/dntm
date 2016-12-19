from collections import OrderedDict

import theano
import theano.tensor as TT

from core.utils import safe_grad, global_rng, block_gradient, as_floatX, \
        safe_izip, sharedX

import numpy as np

class MinibatchGradPartitioner(object):

    def __init__(self, params, sub_mb_size, batch_size, seq_len, nvalidation=360):
        assert batch_size % sub_mb_size == 0, (" batch size  should be divisible"
                                               " by the  sub minibatch size.")
        self.gs = []
        self.gs_mon = []

        self.params = params
        self.__init_vals()
        self.sub_mb_size = sub_mb_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.nvalidation = nvalidation
        self.nbatches = self.batch_size / self.sub_mb_size
        self.updates = OrderedDict({})

    def __init_vals(self):
        self.gs = [theano.shared(as_floatX(k.get_value(borrow=True) * 0.0), \
                        name="grad_%s" % n) for n, k in \
                        self.params.__dict__['params'].iteritems()]

        self.gs_mon = [theano.shared(as_floatX(k.get_value(borrow=True) * 0.0), \
                        name="grad_%s_mon" % n) for n, k in \
                        self.params.__dict__['params'].iteritems()]

    def reset_vals(self):
        for g in self.gs:
            g.set_value(0.0 * g.get_value())

    def construct_updates(self, grads):
        if not self.updates:
            self.updates = OrderedDict({})

        ngrads = OrderedDict({})
        mb_step = sharedX(0, name="mb_step")
        self.updates[mb_step] = mb_step + 1
        cond = TT.eq((mb_step) % self.nbatches, 0)
        rate = 1.0 / self.nbatches

        for op, og in grads.iteritems():
            for i, g in enumerate(self.gs):
                if op.name in g.name:
                    break
            else:
                raise ValueError("Gradient for %s was not found." % op.name)

            if rate < 1.0:
                new_grad = (og + self.gs[i]) * as_floatX(rate)
                self.updates[self.gs[i]] = cond * new_grad + (1 - cond) * og * \
                        as_floatX(rate)
                ngrads[op] = new_grad
            else:
                ngrads[op] = og

        return ngrads

    def get_compute_grad_fn(self, grads, updates, inps):
        ngrads = self.construct_updates(grads)
        self.updates.update(updates.copy())
        compute_grad_fn = theano.function(inps + [self.seq_len],
                                          [],
                                          updates=self.updates,
                                          name="compute_grad_fn")
        return compute_grad_fn, ngrads

    def accum_grads(self, compute_grad_fn, fn_inps):
        def ret_sub_mb_inps(x, y, fn_inps):
            new_inps = OrderedDict({})
            for k, inp in fn_inps.iteritems():
                if isinstance(inp, np.ndarray):
                    if inp.ndim == 2:
                        new_inps[k] = inp[:, x:y]
                    elif inp.ndim == 3 and k != "X":
                        new_inps[k] = inp[:, x:y, :]
                    elif inp.ndim == 3:
                        new_inps[k] = inp[:, :, x:y]
                else:
                    new_inps[k] = inp
            return new_inps

        if self.batch_size != self.sub_mb_size:
            for i in xrange(0, self.batch_size, self.sub_mb_size):
                sub_mb_inps = ret_sub_mb_inps(i, i + self.sub_mb_size, fn_inps)
                xlen = sub_mb_inps['y'].shape[0]
                sub_mb_inps.update({"seq_len": xlen})
                compute_grad_fn(**sub_mb_inps)
        else:
            xlen = fn_inps['y'].shape[0]
            sub_mb_inps.update({"seq_len": xlen})
            compute_grad_fn(**sub_mb_inps)

