import logging
import sys

import numpy

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


EPS = 1e-6
DEFAULT_SEED = 3


global_rng = numpy.random.RandomState(DEFAULT_SEED)
global_trng = RandomStreams(DEFAULT_SEED)


class SEEDSetter(object):

    def __init__(self, file_=""):
        global DEFAULT_SEED
        global global_rng
        global global_trng

        self.file_ = file_

        with open(self.file_, "rb") as fh:
            for line in fh:
                print line
                if "seed" in line:
                    toks = line.split("=")
                    seed = toks[-1].strip()
                    DEFAULT_SEED = int(seed)

        global_rng = numpy.random.RandomState(DEFAULT_SEED)
        global_trng = RandomStreams(DEFAULT_SEED)

    def __str__(self):
        return "value is @ %s" % DEFAULT_SEED


floatX = theano.config.floatX
Sigmoid = lambda x, use_noise=0: TT.nnet.sigmoid(x)
Softmax = lambda x : TT.nnet.softmax(x)
Tanh = lambda x, use_noise=0: TT.tanh(x)
Linear = lambda x: x

#Rectifier nonlinearities
Rect = lambda x, use_noise=0: 0.5 * (x + abs(x + 1e-4))


Leaky_Rect = lambda x, leak=0.95, use_noise=0: ((1 + leak) * x + (1 - leak) * abs(x)) * 0.5
Trect = lambda x, use_noise=0: Rect(Tanh(x + EPS))
Trect_dg = lambda x, d, use_noise=0: Rect(Tanh(d*x))

Softmax = lambda x: TT.nnet.softmax(x)
Linear = lambda x: x

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Change this to change the location of parent directory where your models will be
# dumped into.
SAVE_DUMP_FOLDER="./"

