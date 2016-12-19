import warnings
import theano
import numpy as np
from collections import OrderedDict
import cPickle as pkl

import inspect

from utils.utils import (sample_weights_orth,
                   sample_weights_classic,
                   sample_weights_uni_xav,
                   sample_weights_uni_rect)


from utils import sharedX
from commons import floatX, DEFAULT_SEED

from enum import Enum

InitMethods = Enum("InitMethods",
                   "Orthogonal Classic Adaptive AdaptiveId UniXav UniRect AdaptiveUniXav")


BiasInitMethods = Enum("InitMethods",
                       "Random Zeros Constant")


class Initializer(object):
    def __init__(self,
                 sparsity=-1,
                 scale=0.01,
                 init_method=None,
                 rng=None,
                 center=0.0):

        if init_method is None:
            raise ValueError("init_method should not be empty.")

        if rng is None:
            warnings.warn("rng for the is empty, we are creating a new one.")
            rng = np.random.RandomState(DEFAULT_SEED)

        self.rng = rng
        self.scale = scale
        self.sparsity = sparsity
        self.init_method = init_method
        self.center = center


class BiasInitializer(Initializer):

    def __call__(self, ndim, init_bias_val=1e-6):
        if self.init_method == BiasInitMethods.Random:
            bias = self.rng.uniform(low=-self.scale,
                                    high=self.scale,
                                    size=(ndim,))
        elif self.init_method == BiasInitMethods.Constant:
            if init_bias_val is not None:
                bias = np.zeros((ndim,)) + self.center + init_bias_val
            else:
                bias = np.zeros((ndim,)) + self.center
        return bias


class WeightInitializer(Initializer):

    def __call__(self, n_in, n_out):
        if self.init_method == InitMethods.Orthogonal:
            if n_in != n_out:
                raise ValueError("Number of inputs and outputs should be the same!")
            W = sample_weights_orth(n_in,
                                    self.sparsity,
                                    scale=self.scale,
                                    rng=self.rng)
        elif self.init_method == InitMethods.Classic:
            W = sample_weights_classic(n_in,
                                       n_out,
                                       self.sparsity,
                                       scale=self.scale,
                                       rng=self.rng)
        elif self.init_method == InitMethods.Adaptive:
            if n_in == n_out:
                W = sample_weights_orth(n_in,
                                        self.sparsity,
                                        scale=self.scale,
                                        rng=self.rng)
            else:
                W = sample_weights_classic(n_in,
                                           n_out,
                                           self.sparsity,
                                           scale=self.scale,
                                           rng=self.rng)
        elif self.init_method == InitMethods.AdaptiveId:
            if n_in == n_out:
                Id = np.eye(n_in).astype(floatX)
                W = Id * self.scale
            else:
                W = sample_weights_classic(n_in,
                                           n_out,
                                           self.sparsity,
                                           scale=self.scale,
                                           rng=self.rng)
        elif self.init_method == InitMethods.UniXav:
            W = sample_weights_uni_xav(n_in,
                                       n_out,
                                       rng=self.rng)
        elif self.init_method == InitMethods.UniRect:
            W = sample_weights_uni_rect(n_in,
                                        n_out,
                                        rng=self.rng)
        elif self.init_method == InitMethods.AdaptiveUniXav:
            if n_in == n_out:
                W = sample_weights_orth(n_in,
                                        self.sparsity,
                                        scale=self.scale,
                                        rng=self.rng)
            else:
                W = sample_weights_uni_xav(n_in,
                                           n_out,
                                           rng=self.rng)

        return W


class Parameters(object):
    """
        A class for representing the parameters
        of a model.
    """
    def __init__(self):
        self.__dict__['params'] = OrderedDict({})

    def get_dict(self):
        return self.__dict__['params']

    def __setattr__(self, name, array):
        params = self.get_dict()
        if name not in params:
            params[name] = sharedX(array,
                                   name=name)
        else:
            print "%s already assigned" % name
            if array.shape != params[name].get_value().shape:
                raise ValueError('The shape mismatch for the new value you want to assign'
                                 'to %s' % name)
            params[name].set_value(np.asarray(
                    array,
                    dtype = theano.config.floatX
                ), borrow=True)

    def __setitem__(self, name, array):
        self.__setattr__(name, array)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __getattr__(self, name):
        params = self.__dict__['params']
        return params[name]

    def __merge_with_other(self, other):
        other_dict = other.get_dict()
        current_dict = self.get_dict()

        new_p = Parameters()
        new_p.update(other_dict)
        new_p.update(current_dict)
        return new_p

    def __add__(self, other):
        """
        Add two parameter classes.
        """
        if other == 0:
            return self.__radd__(other)
        else:
            new_p = self.__merge_with_other(other)
            return new_p

    def __radd__(self, other):
        """
        This function is for sum(.)
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def remove(self, name):
        del self.__dict__['params'][name]

    @property
    def values(self):
        params = self.get_dict()
        return params.values()

    @property
    def params_list(self):
        return self.values

    @property
    def keys(self):
        params = self.get_dict()
        return params.keys()

    def save(self, filename):
        params = self.get_dict()
        pkl.dump({p.name:p.get_value() for p in params.values()},
                open(filename, 'wb'), 2)

    def update(self, new_val):
        if isinstance(new_val, Parameters):
            new_dict = new_val.get_dict()
        else:
            if 'params' in new_val:
                new_dict = new_val['params']
            else:
                new_dict = new_val
        return self.__dict__['params'].update(new_dict)

    def init_from_dict(self, new_dict):
        self.__dict__['params'] = new_dict
        return self

    def filterby(self, attrname):
        filtered_attrs = OrderedDict({k: p for k, p in \
                self.__dict__['params'].iteritems() if
            attrname in p.name})
        filtered_params = Parameters().init_from_dict(filtered_attrs)
        return filtered_params

    def lfilterby(self, attrname):
        filtered_params = [p for k, p in self.__dict__['params'].iteritems() if
                          attrname in p.name]
        return filtered_params

    def getparamname(self, attrname):
        name = None
        for k, p in self.__dict__['params'].iteritems():
            if attrname in p.name:
                name = k
                break
        return name

    def load(self, filename):
        loaded = pkl.load(open(filename,'rb'))
        self.set_values(loaded)

    def set_values(self, other):
        params = self.__dict__['params']

        if isinstance(other, Parameters):
            other = other.__dict__['params']

        if not all([len(set(params.keys())) == len(params.keys()),
                    len(set(other.keys())) == len(other.keys()),
                    len(params.keys()) == len(other.keys())]):
            raise ValueError("There is a problem with the shape of the parameters")

        [params[k].set_value(v.get_value()) if hasattr(v, 'get_value') else \
                params[k].set_value(v) \
                for k, v in other.iteritems()]

    def total_nparams(self):
        params = self.get_dict()
        nparams = sum(np.prod(p.get_value().shape) for _, p in params.iteritems())
        return nparams

    def print_param_norms(self):
        params = self.get_dict()
        for k, v in params.iteritems():
            print "param name: %s, param norm: %.2f " % (k, \
                    np.sqrt(((v.get_value())**2).sum()))

    def renormalize_params(self, nscale=5.4):
        params = self.__dict__['params']
        total_norm = 0
        for k, v in params.iteritems():
            total_norm += np.sqrt((v.get_value()**2).sum())
        rho = nscale / total_norm

        print "Rho is, ", rho
        for k, v in params.iteritems():
            v.set_value(v.get_value()*rho)

    def __enter__(self):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        self.__dict__['_env_locals'] = env_locals.keys()

    def __exit__(self, type, value, traceback):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        prev_env_locals = self.__dict__['_env_locals']
        del self.__dict__['_env_locals']

        for k in env_locals.keys():
            if k not in prev_env_locals:
                self.__setattr__(k, env_locals[k])
                env_locals[k] = self.__getattr__(k)
        return True

if __name__ == "__main__":
    P = Parameters()
    test = np.zeros((5,))
    print P.values
