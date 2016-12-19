#!/bin/bash -x

THEANO_FLAGS="floatX=float32,device=cpu,profile=True,profile_optimizer=True" python train.py --exp proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_bowout_qmask #&> proto_ntm_fb_BABI_task_grucontroller_curriculum_simple_small_10k_hard_task2_bowout_qmask_after.txt
