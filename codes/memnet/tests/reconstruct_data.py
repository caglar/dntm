from fbABIdataiterator_fbway2 import FBbABIDataIteratorE2E
import numpy as np

def print_vals(X, ivocab):
    for i in xrange(X.shape[1]):
        if np.all(X[:, i] == 0):
            break
        else:
            str_ = []
            for j in xrange(X[:, i].shape[0]):
                str_.append(ivocab[X[j, i]])
            print " ".join(str_)

seed = 5
bs = 500
rng = np.random.RandomState(seed)

path = "/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en/"
fbbabidataiterator = FBbABIDataIteratorE2E(batch_size=bs,
                                           rng = rng,
                                           seed = 7,
                                           task_path=path,
                                           task_id=9,
                                           randomize=True,
                                           task_file="all_tasks_train_ngram_False.pkl",
                                           fact_vocab="all_tasks_train_ngram_False_dict.pkl")

vocab = fbbabidataiterator.vocab
ivocab = {k:v for v, k in vocab.iteritems()}
j = 0
map_q = lambda sent: " ".join(map(lambda x: ivocab[x], sent))

for batch in fbbabidataiterator:
    task_ids = batch['task_ids']
    for i in xrange(batch['x'].shape[2]):
        print_vals(batch['x'][:, :, i], ivocab)
        print "Question: "
        print map_q(batch['q'][:, i])
        print "Answer: "
        print ivocab[batch['y'][i]]
        print "-----------------------------"
    j += 1

    if j == 1:
        import ipdb; ipdb.set_trace()


