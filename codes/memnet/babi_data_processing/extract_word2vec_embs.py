import cPickle as pkl
import argparse
from sklearn.decomposition import PCA
from collections import OrderedDict, defaultdict
import numpy as np
from itertools import izip

parser = argparse.ArgumentParser(description="Arguments to parse for word2vec embeddings")
parser.add_argument("--vocab", required=True)
parser.add_argument("--w2v-dir", required=True)
parser.add_argument("--emb-size", required=True)
parser.add_argument("--nwords-pca", required=True)
parser.add_argument("--save-dir", required=True)

args = parser.parse_args()
vocab = pkl.load(open(args.vocab, "rb"))
w2v = pkl.load(open(args.w2v_dir, "rb"))
emb_size = int(args.emb_size)
nwords_pca = int(args.nwords_pca)
np.random.seed(2)

nwords_w2v = len(w2v.keys())
w2v_idxs = np.random.permutation(np.arange(nwords_w2v))[nwords_pca]
selected_keys = w2v.keys()[w2v_idxs]
selected_values = [w2v[k] for k in selected_keys]
import ipdb; ipdb.set_trace()

idx2word = OrderedDict({v:k for k, v in vocab.items()})
#idx2word = dict({v:k for k, v in w2v.items()})
nkeys = len(w2v.items())

def prune_embeddings():
    new_embs = defaultdict()
    for k, v in idx2word.items():
        if v in w2v.keys():
            new_embs[k] = w2v[v]
        else:
            rnd_key = np.random.random_integers(0, nkeys-1)
            v = w2v.keys()[rnd_key]
            new_embs[k] = w2v[v]
    return new_embs

w2v_embs = prune_embeddings()
emb_values = np.concatenate([w2v_embs.values(), selected_values], axis=0)
pca  = PCA(n_components=emb_size, whiten=False)
pca.fit(emb_values)
pca_w2v_embs = pca.transform(np.asarray(w2v_embs.values()))
w2v_vocab_map = OrderedDict({k:v for k, v in izip(vocab.values(), pca_w2v_embs)})
pkl.dump(w2v_vocab_map, open(args.save_dir, "wb"))
