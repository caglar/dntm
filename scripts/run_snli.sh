#!/bin/bash -e

echo 'launching the experiment'
echo `hostname`
export PYTHONPATH="/u/gulcehrc/Experiments/codes/python/dntm/codes/":$PYTHONPATH
export THEANO_FLAGS="floatX=float32,device=gpu,force_device=True,lib.cnmem=0.92"
echo $PYTHONPATH

python -m ipdb run_ntm_snli.py
