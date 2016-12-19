#!/bin/bash

export CODE_ROOT=/rap/jvb-000-aa/data/sarath/code/MemoryNetwork/codes/caglar/memnet

echo "launching "
echo $1

OUTPUT=`msub -F "\"$1 0 $2\"" ${CODE_ROOT}/submit_single.sh`

OUTPUT=`echo $OUTPUT | tr -d " "`

echo $OUTPUT



for i in `seq 2 5`;

do

OUTPUT=`msub -F "\"$1 1 $2\"" -l depend=${OUTPUT} ${CODE_ROOT}/submit_single.sh`

OUTPUT=`echo $OUTPUT | tr -d " "`

echo $OUTPUT

done

