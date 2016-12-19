#!/bin/bash -e 

GPU=$1
export PYTHONPATH=../codes
THEANO_FLAGS="floatX=float32,device=${GPU},force_device=True,lib.cnmem=0.85,exception_verbosity='high'" python -m ipdb train_ntm_adam_on_associative_recall.py
