#!/bin/bash -x

smart-dispatch -t 6:00:00 -q gpu_1 -r --pbsFlags="-lfeature=k80 -Ajvb-000-ag" -g 1 -f commands_babi_ff_ntm.txt launch
