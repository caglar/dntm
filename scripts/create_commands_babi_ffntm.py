import numpy

numpy.random.seed(1)
ppath="export PYTHONPATH=/home/gulcehre/code/python/dntm/codes/:$PYTHONPATH; "
save_path = "/rap/jvb-000-aa/gulcehre/dntm/ff_exps/"
save_file = "commands_babi_ff_ntm.txt"

command_template = \
"""THEANO_FLAGS='floatX=float32,device=gpu0,force_device=True,lib.cnmem=0.92,mode=FAST_RUN' python -u submit_ff_soft_nodb.py --seed {0} --save_path {1} --seed {2}"""

command_template = ppath + command_template
cmds = []

njobs = 24
rnd_int = numpy.random.randint

task_ids = numpy.arange(20)
seeds = numpy.arange(5, 15)

for task_id in task_ids:
    for seed in seeds:
        cmds.append(command_template.format(seed, save_path, task_id))

try:
    with open(save_file, "w") as file:
        for cmd in cmds:
            file.write("%s\n" % cmd)
except:
    print "Couldn't write into the file."
