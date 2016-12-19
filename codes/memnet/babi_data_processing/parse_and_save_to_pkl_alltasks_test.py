import os
import cPickle as pkl
import numpy as np
from collections import OrderedDict

PATH = "./"
TASK_NAME = "qa1_single-supporting-fact_train.txt.tok"""
Data = OrderedDict({})

task_id = 0
mode = "test"
vocab = pkl.load(open("all_taskstrain_dict.pkl", "rb"))

def open_file(filename):
    try:
        fileh = open(filename, "r")
    except:
        raise IOError("can't read the file")
    return fileh

def remove_integers(toks):
    ntoks = []

    for tok in toks:
        if not tok.isdigit():
            ntoks.append(tok)

    return ntoks

def add_token_to_vocab(tokens):
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = vocab['nev']
            vocab['nev'] += 1

def map_vocab(tokens):
    fn = lambda x: vocab[x]
    mapped = map(fn, tokens)
    return mapped

def get_task_key():
    return "task_%d" % task_id

def add_to_facts(set_id, qa_id, fact_id, fact_toks):
    task_key = get_task_key()
    if task_key not in Data.keys():
        Data[task_key] = OrderedDict({})

    qa_key = "qa_%d" % qa_id
    if qa_key not in Data[task_key].keys():
        Data[task_key][qa_key] = OrderedDict({})

    set_key = "set_%d" % set_id
    if set_key not in Data[task_key][qa_key].keys():
        Data[task_key]["qa_%d" % qa_id]["set_%d" % set_id] = OrderedDict({})

    add_token_to_vocab(fact_toks)
    Data[task_key]["qa_%d" % qa_id]["set_%d" % set_id]["fact_%d" % fact_id] = map_vocab(fact_toks)

def add_to_qs(set_id, qa_id, q_id, q_toks):
    task_key = get_task_key()
    if task_key not in Data.keys():
        Data[task_key] = OrderedDict({})

    qa_key = "qa_%d" % qa_id
    if qa_key not in Data[task_key].keys():
        Data[task_key]["qa_%d" % qa_id] = OrderedDict({})

    set_key = "set_%d" % set_id
    if set_key not in Data[task_key]["qa_%d" % qa_id].keys():
        Data[task_key]["qa_%d" % qa_id]["set_%d" % set_id] = OrderedDict({})

    add_token_to_vocab(q_toks)
    Data[task_key]["qa_%d" % qa_id]["set_%d" % set_id]["q_%d" % q_id] = map_vocab(q_toks)

def add_to_ans(set_id, qa_id, ans_id, ans_toks):
    task_key = get_task_key()
    if task_key not in Data.keys():
        Data[task_key] = OrderedDict({})

    ans_len = len(ans_toks) // 2
    ans_toks = ans_toks[:ans_len]
    qa_key = "qa_%d" % qa_id

    if qa_key not in Data[task_key].keys():
        Data[task_key]["qa_%d" % qa_id] = OrderedDict({})

    set_key = "set_%d" % set_id

    if set_key not in Data[task_key][qa_key].keys():
        Data[task_key]["qa_%d" % qa_id][set_key] = OrderedDict({})

    ans_toks = remove_integers(ans_toks)
    ans_tok_len = len(ans_toks)
    for i in xrange(ans_tok_len):
        add_token_to_vocab(ans_toks[i])
        Data[task_key][qa_key][set_key]["ans_%d" % (ans_id + i)] = map_vocab(ans_toks[i])
    return ans_tok_len

def read_files(fileh):
    qa_c = -1
    set_c = 0
    fact_c = 0
    q_c = 0
    ans_c = 0

    prev_q = False
    for line in fileh:
        tokens = line.rstrip().split(" ")
        idx = tokens[0]
        tokens = tokens[1:]
        if int(idx) == 1:
            set_c = 0
            fact_c = 0
            q_c = 0
            ans_c = 0
            prev_q = False
            qa_c += 1
            add_to_facts(set_c, qa_c, fact_c, tokens)
            fact_c += 1
        else:
            if "?" in tokens:
                prev_q = True
                q_end_idx = tokens.index("?")
                ans_start_idx = q_end_idx + 1
                add_to_qs(set_c, qa_c, q_c, tokens[:ans_start_idx])
                ainc = add_to_ans(set_c, qa_c, ans_c, tokens[ans_start_idx:])
                q_c += 1
                ans_c += ainc
            else:
                if prev_q:
                    set_c += 1
                add_to_facts(set_c, qa_c, fact_c, tokens)
                fact_c += 1
                prev_q = False

task_files = sorted(os.listdir(PATH))

task_indices = []
task_files_mode = []
for task in task_files:
    if mode in task and "all_tasks" not in task and task.endswith("tok"):
        prefix = task.split("_")[0]
        task_indices.append(int("".join(prefix[2:])))
        task_files_mode.append(task)

sorted_indices = np.argsort(task_indices)
n_task_files = map(lambda x: task_files_mode[x], sorted_indices)

for fn in n_task_files:
    if os.path.isfile(fn):
        if fn.endswith("tok") and "all_tasks" not in fn:
            print "Saving the file, ", fn
            fileh = open_file(PATH + fn)
            read_files(fileh)
            fileh.close()
            task_id += 1

dump_fileh = open(PATH + "all_tasks" + mode + ".pkl", "wb")
pkl.dump(Data, dump_fileh)
dump_fileh.close()

dump_fileh_vocab = open(PATH + "all_tasks" + mode  + "_dict.pkl", "wb")
pkl.dump(vocab, dump_fileh_vocab)
dump_fileh_vocab.close()

