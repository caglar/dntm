#!/bin/bash -x

smart-dispatch -q gpu_1 -t 22:20:00 --pbsFlags="-lfeature=k80 -Ajvb-000-ag" -g 1 -f commands_dntm_res_ar_v0.txt launch
