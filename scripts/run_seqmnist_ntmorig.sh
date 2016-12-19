#!/bin/bash -e

FUEL_DATA_PATH=/raid/gulcehrc/data/ \
    PYTHONPATH=/home/gulcehrc/Codes/python/tardis/tardis/:$PYTHONPATH \
    THEANO_FLAGS="floatX=float32,device=gpu3,force_device=True,lib.cnmem=0.92" python -m ipdb submit_single_seqmnist_ntm_orig.py
