#!/bin/bash -x

NAME=$1

if [[ -n $2 ]]; then
NJOBS=$2
else   
    echo "ERROR: No argument is supplied for the number of jobs."
    exit
fi

if [[ -n "$NAME" ]]; then
    jobdispatch  --raw="#PBS -l feature=k80" --repeat_jobs=$NJOBS --gpu --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True,lib.cnmem=0.85 jobman sql "postgresql://gulcehrc@opter.iro.umontreal.ca/gulcehrc_db?table=${NAME}" .
else
    echo "ERROR: No argument is supplied for the table name."
fi
