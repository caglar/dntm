from fbABIdataiterator_singleq import FBbABIDataIteratorSingleQ
import numpy as np

def print_vals(X, ivocab):
    for i in xrange(X.shape[1]):
        if np.all(X[:, i] == 0):
            break
        else:
            str_ = []
            for j in xrange(X[:, i].shape[0]):
                if ivocab[X[j, i]] != ' ' and ivocab[X[j, i]] != '':
                    str_.append(ivocab[X[j, i]])

            print " ".join(str_)

seed = 5
bs = 500

rng = np.random.RandomState(seed)
task_id = 2 #20 #17
max_fact_len = 12 #12
max_seq_len = 120 #15 #4

path = "/raid/gulcehrc/tasks_1-20_v1-2/en-10k/"

fbbabidataiterator = FBbABIDataIteratorSingleQ(batch_size=bs,
                                               rng = rng,
                                               seed = 5,
                                               task_path=path,
                                               task_id=task_id,
                                               randomize=False,
                                               max_fact_len=max_fact_len,
                                               max_seq_len=max_seq_len,
                                               task_file="all_tasks_test_ngram_False.pkl",
                                               fact_vocab="all_tasks_test_ngram_False_dict.pkl")

vocab = fbbabidataiterator.vocab
ivocab = {k:v for v, k in vocab.iteritems()}
j = 0
map_q = lambda sent: " ".join(map(lambda x: ivocab[x], sent))

for batch in fbbabidataiterator:
    task_ids = batch['task_ids']
    import ipdb; ipdb.set_trace()

    for i in xrange(batch['x'].shape[2]):
        if batch['y'][:, i].nonzero()[0] != batch['cmask'][:, i].nonzero()[0]:
            import ipdb; ipdb.set_trace()
    j += 1

