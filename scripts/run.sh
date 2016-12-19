#!/bin/bash -x
GPU=$1
THEANO_FLAGS="floatX=float32,device=${GPU},force_device=True,exception_verbosity='high',lib.cnmem=0.85" python -m ipdb train.py --exp $2

