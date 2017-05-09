#!/bin/bash

#PBS -l nodes=1:gpus=1
#PBS -l walltime=08:00:00
#PBS -A jvb-000-ag
#PBS -m bea
#PBS -l feature=k80

# Use msub on helios1 to submit this.
#

# msub ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/04_race/launch_010.sh
# msub -l depend=51777 ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/04_race/launch_010.sh


#OUTPUT=`msub ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/04_race/launch_010.sh`
#OUTPUT=`echo $OUTPUT | tr -d " "`
#echo $OUTPUT

#OUTPUT=`msub -l depend=${OUTPUT} ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/04_race/launch_010.sh`
#OUTPUT=`echo $OUTPUT | tr -d " "`
#echo $OUTPUT
export CODE_ROOT=/home/gulcehre/code/python/MemoryNetwork/codes/caglar/memnet/

$t1 = "--task_id"
$t2 = "--reload_model"
$t3 = "--save_path"

THEANO_FLAGS=device=gpu,floatX=float32,force_device=True,lib.cnmem=0.88 python -u ${CODE_ROOT}/submit_ff_soft_nodb.py --task_id $1 --reload_model $2 --save_path $3
