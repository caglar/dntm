#!/bin/bash -e

export PYTHONPATH="/u/gulcehrc/Experiments/codes/python/dntm/codes/":$PYTHONPATH
export THEANO_FLAGS="floatX=float32,device=gpu,force_device=True,lib.cnmem=1"
python -m pdb run_models_adam_grucont_soft_single.py
