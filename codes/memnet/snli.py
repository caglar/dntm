import theano

import os
import cPickle as pkl
import numpy
import warnings
from six import Iterator

numpy.random.seed(3)


class SNLI(Iterator):

    def __init__(self,
                 batch_size=50,
                 loadall=False,
                 datapath=None,
                 random_flip_order=False,
                 mode="train"):

        self.batch_size = batch_size
        self.datapath = datapath
        assert mode in ["train", "test", "valid"]
        self.mode = mode

        data_file = open(self.datapath, 'rb')
        print "Loading the dataset in mode, ", mode
        data_dict = pkl.load(data_file)
        self.train_set, self.dev_set, self.test_set = data_dict['train_valid_test']

        self.train_size = len(self.train_set)
        self.dev_size = len(self.dev_set)
        self.test_size = len(self.test_set)
        self.random_flip_order = random_flip_order

        self.train_idxs = numpy.arange(self.train_size)
        self.shuffled_train_idxs = numpy.random.permutation(self.train_idxs)

        if mode == "train":
            del self.test_set
            del self.dev_set

        if mode == "dev":
            del self.train_set
            del self.test_set

        if mode == "test":
            del self.dev_set
            del self.train_set


        self.w_referred = data_dict['w_referred']
        data_file.close()
        self.train_ptr = 0
        self.dev_ptr = 0
        self.test_ptr = 0

    def __iter__(self):
        return self

    def train_minibatch_generator(self):
        # 2 is added to the indices of the hypo and the premise in order to
        # add the end of premise and end of hypothesis tags.
        while self.train_ptr <= self.train_size - 1:
            self.train_ptr += self.batch_size
            idxs = self.shuffled_train_idxs[self.train_ptr - self.batch_size : self.train_ptr]
            flipped = []

            if not self.random_flip_order:
                minibatch = [(map(lambda x: x + 2, self.train_set[idx][0]),
                            map(lambda x: x + 2, self.train_set[idx][1]),
                            self.train_set[idx][2])  for idx in idxs]
                flipped = [0 for i in xrange(self.batch_size)]
            else:
                minibatch = []
                for idx in idxs:
                    y = self.train_set[idx][2]
                    if y in [1, 2]:
                        if numpy.random.uniform() > 0.5:
                            hypo = self.train_set[idx][0]
                            premise = self.train_set[idx][1]
                            flipped.append(1)
                        else:
                            hypo = self.train_set[idx][1]
                            premise = self.train_set[idx][0]
                            flipped.append(0)
                    else:
                        hypo = self.train_set[idx][1]
                        premise = self.train_set[idx][0]
                        flipped.append(0)
                    minibatch.append((premise, hypo, y))

            mblen = len(minibatch)

            if mblen < self.batch_size:
                remaining = self.batch_size - mblen
                ridxs = self.shuffled_train_idxs[:remaining]
                fill_exs = [(map(lambda x: x + 2, self.train_set[idx][0]),
                             map(lambda x: x + 2, self.train_set[idx][1]),
                             self.train_set[idx][2]) for idx in ridxs]
                minibatch += fill_exs

            longest_premise, longest_hypo = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            X = numpy.zeros((self.batch_size, longest_hypo + longest_premise + 2), dtype="uint32")
            mask = numpy.zeros((self.batch_size, longest_hypo + longest_premise + 2), dtype="float32")
            y = numpy.zeros((self.batch_size,), dtype='uint8')

            for i, (p, h, t) in enumerate(minibatch):
                is_flipped = flipped[i]
                len_h = len(h)
                len_p = len(p)
                X[i, :len_p] = p
                X[i, len_p] = 1. if is_flipped else 0.
                X[i, len_p + 1:(len_h + len_p + 1)] = h
                X[i, len_h + len_p + 1] = 0. if is_flipped else 1.
                mask[i, :len_h + len_p + 2] = 1.
                y[i] = t
            return X.T, mask.T, y
        else:
            self.train_ptr = 0
            self.shuffled_train_idxs = numpy.random.permutation(self.train_idxs)
            raise StopIteration

    def dev_minibatch_generator(self):
        # 2 is added to the indices of the hypo and the premise in order to
        # add the end of premise and end of hypothesis tags.
        while self.dev_ptr <= self.dev_size - self.batch_size:
            self.dev_ptr += self.batch_size
            minibatch = self.dev_set[self.dev_ptr - self.batch_size : self.dev_ptr]
            minibatch = [(map(lambda x: x + 2, el[0]), map(lambda x: x + 2, el[1]),
                          el[2]) for el in minibatch]

            if len (minibatch) < self.batch_size:
                warnings.warn("There will be empty slots in minibatch data.", UserWarning)

            longest_premise, longest_hypo = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            X = numpy.zeros((self.batch_size, longest_hypo + longest_premise + 2), dtype="uint32")
            mask = numpy.zeros((self.batch_size, longest_hypo + longest_premise + 2), dtype="float32")
            y = numpy.zeros((self.batch_size,), dtype='uint8')

            for i, (p, h, t) in enumerate(minibatch):
                len_h = len(h)
                len_p = len(p)
                X[i, :len_p] = p
                X[i, len_p + 1:(len_h + len_p + 1)] = h
                X[i, len_h + len_p + 1] = 1
                mask[i, :len_h + len_p + 2] = 1.
                y[i] = t
            return X.T, mask.T, y
        else:
            self.dev_ptr = 0
            raise StopIteration

    def test_minibatch_generator(self):
        # 2 is added to the indices of the hypo and the premise in order to
        # add the end of premise and end of hypothesis tags.
        while self.test_ptr <= self.test_size - self.batch_size:
            self.test_ptr += self.batch_size
            minibatch = self.test_set[self.test_ptr - self.batch_size : self.test_ptr]
            minibatch = [(map(lambda x: x + 2, el[0]), map(lambda x: x + 2, el[1]),
                          el[2]) for el in minibatch]

            if len (minibatch) < self.batch_size:
                warnings.warn("There will be empty slots in minibatch data.", UserWarning)

            longest_premise, longest_hypo = \
                numpy.max(map(lambda x: (len(x[0]), len(x[1])), minibatch), axis=0)

            X = numpy.zeros((self.batch_size, longest_hypo + longest_premise + 2), dtype="uint32")
            mask = numpy.zeros((self.batch_size, longest_hypo + longest_premise + 2), dtype="float32")
            y = numpy.zeros((self.batch_size,), dtype='uint8')

            for i, (p, h, t) in enumerate(minibatch):
                len_h = len(h)
                len_p = len(p)
                X[i, :len_p] = p
                X[i, len_p + 1:(len_h + len_p + 1)] = h
                X[i, len_h + len_p + 1] = 1
                mask[i, :len_h + len_p + 2] = 1.
                y[i] = t
            return X.T, mask.T, y
        else:
            self.test_ptr = 0
            raise StopIteration

    def next(self):
        if self.mode == "train":
            return self.train_minibatch_generator()
        elif self.mode == "valid":
            return self.dev_minibatch_generator()
        elif self.mode == "test":
            return self.test_minibatch_generator()


if __name__ == "__main__":
    snli = SNLI(datapath="/data/lisatmp4/gulcehrc/data/snli/SNLI_data.pkl")
    batch = next(snli)
    import ipdb; ipdb.set_trace()
