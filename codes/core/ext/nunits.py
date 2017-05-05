"""
Maintainer: Caglar Gulcehre
E-mail: ca9lar (at) gmail <dot> com
------------------------------------

You can find the different noisy activations functions  proposed in the  Noisy
Activation Functions paper.

"""


import logging

import theano
import theano.tensor as T
import sys
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from core.commons import DEFAULT_SEED as DSEED

EPS = 1e-6
floatX = theano.config.floatX


global_rng = numpy.random.RandomState(DSEED)
global_trng = RandomStreams(DSEED)


Sigmoid = lambda x, use_noise=0: T.nnet.sigmoid(x)
HardSigmoid = lambda x, angle=0.25: T.maximum(T.minimum(angle*x + 0.5,
                                                          1.0), 0.0)

lin_sigmoid = lambda x: 0.25 * x + 0.5
HardTanh = lambda x: T.minimum(T.maximum(x, -1.), 1.)
HardSigmoid = lambda x: T.minimum(T.maximum(lin_sigmoid(x), 0.), 1.)


Tanh = lambda x: T.tanh(x)
Linear = lambda x: x
Rect = lambda x: T.nnet.relu(x)


floatX = theano.config.floatX
sigmoid = lambda x: T.nnet.sigmoid(x)
tanh = lambda x: T.tanh(x)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('Noisy Units Logger')
logger.addHandler(logging.StreamHandler())


def NHardTanh(x,
              use_noise=1,
              c=0.05):
    """
    Noisy Hard Tanh Units: NANI as proposed in the paper
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """
    logger.info("c: %f" % c)
    threshold = 1.001
    noise = global_trng.normal(size=x.shape,
                               avg=0.,
                               std=1.0,
                               dtype=floatX)

    if not use_noise:
        noise = 0.

    res = HardTanh(x + c * noise)
    return res


def NHardSigmoid(x,
                 use_noise=1,
                 c=0.05):
    """
    Noisy Hard Sigmoid Units: NANI as proposed in the paper
    ----------------------------------------------------
    Arguments:

        x: theano tensor variable, input of the function.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """

    logger.info("c: %f" % c)
    noise = global_trng.normal(size=x.shape,
                               avg=0.,
                               std=1.0,
                               dtype=floatX)

    if not use_noise:
        noise = 0.

    res = HardSigmoid(x + c * noise)
    return res

def NReLU(x,
          use_noise=1,
          c=0.05):
    """
    Noisy Rectifier Sigmoid Units: NANI as proposed in the paper
    ----------------------------------------------------
    Arguments:

        x: theano tensor variable, input of the function.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """

    logger.info("c: %f" % c)
    noise = global_trng.normal(size=x.shape,
                               avg=0.,
                               std=1.0,
                               dtype=floatX)

    if not use_noise:
        noise = 0.

    res = Rect(x + c * noise)
    return res


def NHardTanhSat(x,
                 use_noise=1,
                 c=0.25):
    """
    Noisy Hard Tanh Units at Saturation: NANIS as proposed in the paper
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """
    logger.info("c: %f" % c)
    threshold = 1.001
    noise = global_trng.normal(size=x.shape,
                               avg=0.,
                               std=1.0,
                               dtype=floatX)

    if not use_noise:
        noise = 0.

    test = T.cast(abs(x) > threshold, "float32")
    res = test * HardTanh(x + c * noise) + (1. - test) * HardTanh(x)
    return res


def NHardSigmoidSat(x,
                    use_noise=1,
                    c=0.25):
    """
    Noisy Hard Sigmoid Units at Saturation: NANIS as proposed in the paper
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
    """
    logger.info("c: %d" % c)
    threshold = 2.0
    noise = global_trng.normal(size=x.shape,
                               avg=0.,
                               std=1.0,
                               dtype=floatX)

    if not use_noise:
        noise = 0.

    test = T.cast(abs(x) > threshold, "float32")
    res = test * HardSigmoid(x + c * noise) + \
            (1. - test) + HardSigmoid(x)
    return res


def NTanh(x,
          use_noise=1,
          alpha=1.05,
          c=0.5):
    """
    Noisy Hard Tanh Units: NAN without learning p
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: the leaking rate from the linearized function to the nonlinear one.
    """

    logger.info("c: %f" % c)
    threshold = 1.0
    noise = global_trng.normal(size=x.shape,
                               avg=0.,
                               std=1.0,
                               dtype=floatX)

    signs = T.sgn(x)
    delta = abs(x) - threshold

    scale = c * (T.nnet.sigmoid(delta**2) - 0.5)**2

    if half_normal:
        if alpha > 1.0:
            scale *= -1
        noise = abs(noise)
        if not use_noise:
            noise = 0.797
    elif not use_noise:
        noise = 0.

    eps = scale * noise + alpha * delta
    z = x - signs * eps
    test = T.cast(abs(x) >= threshold, floatX)
    testf = T.cast(test, "float32")
    res = T.switch(test,
                     z,
                     x)
    return res


def NSigmoid(x,
              use_noise=1,
              alpha=1.15,
              c=0.25,
              threshold=2.0):
    """
    Noisy Hard Sigmoid Units: NAN without learning p
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: the leaking rate from the linearized function to the nonlinear one.
    """

    logger.info("c: %f" % c)
    signs = T.sgn(x)
    delta = abs(x) - threshold

    scale = c * (T.nnet.sigmoid(delta**2)  - 0.5)**2

    noise = global_trng.normal(size=x.shape,
                                   avg=0,
                                   std=1.0,
                                   dtype=floatX)

    if half_normal:
       if alpha > 1.0:
           scale *= -1
       noise = abs(noise)
       if not use_noise:
            noise = 0.797
    elif not use_noise:
        noise = 0.

    eps = scale * noise + alpha * delta
    signs = T.sgn(x)
    z = x - signs * eps

    test = T.cast(T.ge(abs(x), threshold), floatX)
    res = test * z + (1. - test) * HardSigmoid(x)

    return res


def NTanhP(x,
           p,
           use_noise=1,
           alpha=1.15,
           c=0.5,
           noise=None,
           clip_output=False,
           half_normal=False):
    """
    Noisy Hard Tanh Units: NAN with learning p
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        p: theano shared variable, a vector of parameters for p.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: float, the leakage rate from the linearized function to the nonlinear one.
        half_normal: bool, whether the noise should be sampled from half-normal or
        normal distribution.
    """


    logger.info("c: %f" % c)
    if not noise:
        noise = global_trng.normal(size=x.shape,
                                   avg=0.,
                                   std=1.0,
                                   dtype=floatX)

    signs = T.sgn(x)
    delta = HardTanh(x) - x

    scale = c * (T.nnet.sigmoid(p * delta) - 0.5)**2
    noise_det = 0.
    if half_normal:
        if alpha > 1.0:
            scale *= -1.
        noise = abs(noise)
        if not use_noise:
            noise_det = numpy.float32(0.797)
    elif not use_noise:
        noise_det = 0.
    noise = use_noise * noise + (1. - use_noise) * noise_det
    res = alpha * HardTanh(x) + (1. - alpha) * x - signs * scale * noise

    if clip_output:
        return HardTanh(res)
    return res


def NSigmoidP(x,
              p,
              use_noise=1,
              alpha=1.1,
              c=0.15,
              noise=None,
              half_normal=True):
    """
    Noisy Sigmoid Tanh Units: NAN with learning p
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        p: theano shared variable, a vector of parameters for p.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: float, the leakage rate from the linearized function to the nonlinear one.
        half_normal: bool, whether the noise should be sampled from half-normal or
        normal distribution.
    """
    lin_sigm = 0.25 * x + 0.5
    logger.info("c: %f" % c)
    signs = T.sgn(x)
    delta = HardSigmoid(x) - lin_sigm
    signs = T.sgn(x)
    scale = c * (T.nnet.sigmoid(p * delta) - 0.5)**2
    if not noise:
        noise = global_trng.normal(size=x.shape,
                                   avg=0,
                                   std=1.0,
                                   dtype=floatX)
    noise_det = 0.
    if half_normal:
       if alpha > 1.0:
          scale *= -1.
       if not use_noise:
           noise_det = numpy.float32(0.797)
       else:
           noise = abs(noise)
    elif not use_noise:
        noise_det = 0.

    noise = use_noise * noise + (1. - use_noise) * noise_det
    res = (alpha * HardSigmoid(x) + (1. - alpha) * lin_sigm - signs * scale * noise)
    return res


def NSigmoidPInp(x,
               p,
               use_noise=1,
               c=0.25,
               half_normal=False):
    """
    Noisy Sigmoid where the noise is injected to the input: NANI with learning p.
    This function works well with discrete switching functions.
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        p: theano shared variable, a vector of parameters for p.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        half_normal: bool, whether the noise should be sampled from half-normal or
        normal distribution.
    """

    logger.info("c: %f" % c)
    signs = T.sgn(x)
    delta = HardSigmoid(x) - (0.25 * x + 0.5)
    signs = T.sgn(x)
    noise = global_trng.normal(size=x.shape,
                               avg=0,
                               std=1.0,
                               dtype=floatX)
    noise_det = 0.
    if half_normal:
       if alpha > 1.0:
          c *= -1
       noise_det = 0.797
       noise = abs(noise)
    elif not use_noise:
        noise = 0.

    noise = use_noise * noise + (1. - use_noise) * noise_det
    scale = c * T.nnet.softplus(p * abs(delta) / (abs(noise) + 1e-10))
    res = HardSigmoid(x + scale * noise)
    return res

def NTanhPInp(x,
              p,
              use_noise=1,
              c=0.25,
              half_normal=False):
    """
    Noisy Tanh units where the noise is injected to the input: NANI with learning p.
    This function works well with discrete switching functions.
    ----------------------------------------------------
    Arguments:
        x: theano tensor variable, input of the function.
        p: theano shared variable, a vector of parameters for p.
        use_noise: int, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        half_normal: bool, whether the noise should be sampled from half-normal or
        normal distribution.
    """

    logger.info("c: %f" % c)
    signs = T.sgn(x)
    delta = HardTanh(x) - x
    signs = T.sgn(x)
    noise = global_trng.normal(size=x.shape,
                               avg=0,
                               std=1.0,
                               dtype=floatX)
    noise_det = 0.
    if half_normal:
       if alpha > 1.0:
          c *= -1
       noise_det = 0.797
       noise = abs(noise)
    elif not use_noise:
        noise = 0.

    noise = use_noise * noise + (1. - use_noise) * noise_det
    scale = c * T.nnet.softplus(p * abs(delta) / (abs(noise) + 1e-10))
    res = HardTanh(x + scale * noise)
    return res

