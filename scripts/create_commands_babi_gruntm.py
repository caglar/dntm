import numpy
import argparse

parser = argparse.ArgumentParser("Parameters for the single soft model.")
parser.add_argument("--task_id", default=1, type=int)

args = parser.parse_args()

numpy.random.seed(1)

ppath="export PYTHONPATH=/rap/jvb-000-aa/data/sarath/code/new-ntm/dntm/codes/:$PYTHONPATH; "

save_path = "/rap/jvb-000-aa/data/sarath/newlog/models/"
save_file = "commands_babi_gru_ntm_taskid_{0}.txt".format(args.task_id)

command_template = \
"""THEANO_FLAGS='floatX=float32,device=gpu0,force_device=True,lib.cnmem=0.92,mode=FAST_RUN' python -u submit_gru_hard_nodb.py --seed {0} --save_path {1} --task_id {2}"""

command_template = ppath + command_template
cmds = []

njobs = 24
rnd_int = numpy.random.randint

seeds = numpy.arange(5, 15)

for seed in seeds:
    cmds.append(command_template.format(seed, save_path, args.task_id))

try:
    with open(save_file, "w") as file:
        for cmd in cmds:
            file.write("%s\n" % cmd)
except:
    print "Couldn't write into the file."
