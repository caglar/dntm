import abc
from abc import ABCMeta
import theano
import theano.tensor as TT

from core.utils import safe_grad, global_rng, block_gradient, as_floatX, \
        safe_izip, sharedX


class Penalty(object):
    def __init__(self, level=None):
        self.level = level

class ParamPenalty(Penalty):
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def penalize_layer_weights(self, layer):
        pass

    @abc.abstractmethod
    def penalize_params(self, param):
        pass

    @abc.abstractmethod
    def get_penalty_level(self):
        pass


class L2Penalty(Penalty):

    def __init__(self, level=None):
        self.level = level
        self.reg = 0.0

    def penalize_layer_weights(self, layer):
        weight = layer.params.filterby("Weight").values[0]
        self.reg += (weight**2).sum()

    def penalize_params(self, param):
        if isinstance(param, list):
            self.reg += sum((p**2).sum() for p in param)
        else:
            self.reg += (param**2).sum()

    def get_penalty_level(self):
        return self.level * self.reg


class L1Penalty(Penalty):

    def __init__(self, level=None):
        self.level = level
        self.reg = 0.0

    def penalize_layer_weights(self, layer):
        weight = layer.params.filterby("Weight").values[0]
        self.reg += abs(weight).sum()

    def penalize_params(self, param):
        if isinstance(param, list):
            self.reg += sum(abs(p).sum() for p in param)
        else:
            self.reg += abs(param).sum()

    def get_penalty_level(self):
        return self.level * self.reg


class WeightNormConstraint(Penalty):
    """
    Add a norm constraint on the weights of a neural network.
    """
    def __init__(self, limit, min_limit=0, axis=1):
        assert limit is not None, (" Limit for the weight norm constraint should"
                                   " not be empty.")
        self.limit = limit
        self.min_limit = min_limit
        self.max_limit = limit
        self.axis = axis

    def __call__(self, updates, weight_name=None):
        weights = [key for key in updates.keys() if key.name == weight_name]
        if len(weights) != 1:
            raise RuntimeError("More than one weight has been found with "
                               " a name for norm constraint.")

        weight = weights[0]
        updated_W = updates[weight]
        l2_norms = TT.sqrt((updated_W**2).sum(axis=self.axis, keepdims=True))
        desired_norms = TT.clip(l2_norms, self.min_limit, self.max_limit)
        scale = desired_norms / TT.maximum(l2_norms, 1e-7)
        updates[weight] = scale * updated_W


class AntiCorrelationConstraint(Penalty):
    """
    Add a norm constraint on the weights of a neural network.
    """
    def __init__(self, level=1e-3, axis=-1):
        self.level = level
        self.axis = axis

    def __call__(self, tens1, tens2, mask):
        if mask.ndim == 2 and tens1.ndim == 4:
            mask = mask.dimshuffle(0, 1, 'x')

        if tens1.ndim != tens2.ndim:
            raise ValueError("Number of dimensions for the first and the second"
                             " tensors should be the same")
        mult_tens = (tens1 * tens2).sum(self.axis)
        #mult_norms = ((tens1**2).sum(self.axis) * (tens2**2).sum(self.axis))**0.5
        #reg = (1 + (mult_tens + 1e-8) / (mult_norms + 1e-8))**2
        reg = mult_tens**2
        reg = self.level * (mask * reg).sum(0).mean()
        return reg


class CorrelationConstraint(Penalty):
    """
    This is to increase the correlation between the read and write weights.
    """
    def __init__(self, level=1e-3, axis=-1):
        self.level = level
        self.axis = axis

    def __call__(self, read_ws, write_ws, mask, qmask):
        if read_ws.ndim == 4:
            read_ws = read_ws.mean(1)

        if read_ws.ndim == 3 and qmask.ndim == 2:
            qmask = qmask.dimshuffle(0, 1, 'x')
            mask = mask.dimshuffle(0, 1, 'x')
        else:
            raise ValueError

        read_ws = read_ws[:, :, 1:]

        if write_ws.ndim == 3 and qmask.ndim == 3:
            write_ws = write_ws[:, :, :-1]
        else:
            raise ValueError

        read_ws *= qmask
        read_ws *= mask
        write_ws *= mask

        summed_readws = read_ws.sum(0) / (qmask.sum(0)[:, None] + 1e-8)
        summed_writews = block_gradient(write_ws.sum(0))
        dot_prods = (summed_readws * summed_writews).sum(self.axis)
        reg = self.level * ((1 - dot_prods)**2).mean()
        return reg


class ReinforcePenalty(Penalty):
    """
    Defines the Reinforce as an additional objective function to the cost. The
    gradient of this is analytically same as doing reinforce.
    """
    def __init__(self, reinf_level, maxent_level, use_reinforce_baseline):
        self.reinf_level = reinf_level
        self.use_reinforce_baseline = use_reinforce_baseline
        self.maxent_level = maxent_level

    def __call__(self, baseline, cost, probs, samples,
                 mask=None, center=None, cost_std=None,
                 child_probs=None,
                 child_samples=None):

        if mask is not None and mask.ndim != probs.ndim:
            mask = mask.dimshuffle(0, 1, 'x')

        if center is None:
            center = 0

        if cost_std is None:
            cost_std = 1

        wsamples = probs * samples
        if self.use_reinforce_baseline:
           reward = cost.dimshuffle('x', 0, 'x')
        else:
            if cost.ndim == 1:
                reward = cost.dimshuffle('x', 0, 'x')
                baseline = baseline.dimshuffle(1, 0, 2)
            else:
                reward = cost.dimshuffle(0, 1, 'x')

        N = probs.shape[-1]
        centered_reward = block_gradient((reward - baseline - center) / cost_std)
        cprobs1 = None

        if child_probs is None or child_samples is None:
            cprobs1 = samples * TT.log(probs.clip(1e-8, 1 - 1e-8))
        else:
            cprobs1 = samples * TT.log(probs.clip(1e-8, 1 - 1e-8)) + \
                    child_samples * TT.log(child_probs.clip(1e-8, 1 - 1e-8))

        if samples.ndim == 4:
            if centered_reward.ndim == 3:
                centered_reward = centered_reward.dimshuffle(0, 'x', 1, 2)
            
            centered_reward = TT.addbroadcast(centered_reward, 3)

            constraint1 = self.reinf_level * ((centered_reward * cprobs1).sum(1))
            constraint2 = (self.maxent_level) * (probs * TT.log(probs.clip(1e-8, \
                    1 - 1e-8))).sum(1)
            policy = (TT.log(probs.clip(1e-8, 1 - 1e-8)) * samples).mean((2, 3)).sum()
        else:
            constraint1 = self.reinf_level * ((centered_reward * cprobs1))
            constraint2 = (self.maxent_level) * (probs * TT.log(probs.clip(1e-8, \
                    1 - 1e-8)))
            policy = (TT.log(probs.clip(1e-8, 1 - 1e-8)) * samples).mean((1, 2)).sum()

        if mask:
            constraint = mask * (constraint1 + constraint2)
        else:
            constraint = (constraint1 + constraint2)

        constraint = constraint.mean((1, 2)).sum()
        return constraint, policy
