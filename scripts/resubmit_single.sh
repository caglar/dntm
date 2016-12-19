#!/bin/bash

export CODE_ROOT=/rap/jvb-000-aa/data/sarath/code/MemoryNetwork/codes/caglar/memnet

echo "launching "
echo $1

OUTPUT=`msub -F "\"$1 1 $2\"" ${CODE_ROOT}/submit_single.sh`

OUTPUT=`echo $OUTPUT | tr -d " "`

echo $OUTPUT
