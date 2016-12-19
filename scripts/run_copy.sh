#!/bin/bash -e 

GPU=$1

THEANO_FLAGS="floatX=float32,device=${GPU},force_device=True,lib.cnmem=0.85" python -m ipdb train_ntm_adam_on_copy.py
