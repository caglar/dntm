#!/bin/bash -x

#THEANO_FLAGS="floatX=float32,device=gpu2,force_device=True,lib.cumem=True" python -m ipdb train.py --exp $1
THEANO_FLAGS="floatX=float32,device=cpu,force_device=True" python -m ipdb train.py --exp $1

