#!/bin/bash -x

THEANO_FLAGS="floatX=float32,device=cpu,force_device=True,exception_verbosity='high',lib.cnmem=0.8" python -m ipdb ../codes/memnet/train.py --exp $1
