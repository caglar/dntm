import numpy

numpy.random.seed(1)
ppath="export PYTHONPATH=/home/gulcehre/code/python/dntm/codes/:$PYTHONPATH; "
command_template = \
"""THEANO_FLAGS='floatX=float32,device=gpu,force_device=True,lib.cnmem=0.92,mode=FAST_RUN' python -u run_dntm_snli.py --lr {0} --nhids {1} --mem-size {2} --batch-size {3} --emb-scale {4}"""

command_template = ppath + command_template
cmds = []

njobs = 30
rnd_int = numpy.random.randint

nhids = [300]
batch_sizes = [32, 64]
emb_scales = [0.8, 0.4, 0.2, 0.1]
mem_sizes = [16, 24, 36, 40]
use_predictive_rewards = [False, True]

for i in xrange(njobs):
    if i % 5 == 0:
        print "job ", i

    lr = 10**(numpy.random.uniform(-1.9, -3.9))

    nhid = nhids[rnd_int(0, len(nhids))]
    batch_size = batch_sizes[rnd_int(0, len(batch_sizes))]
    emb_scale = emb_scales[rnd_int(0, len(emb_scales))]
    mem_size = mem_sizes[rnd_int(0, len(mem_sizes))]

    cmds.append(command_template.format(lr, nhid, mem_size, batch_size, emb_scale))


with open("commands_snli_dntm.txt", "w") as file:
    for cmd in cmds:
        file.write("%s\n" % cmd)
