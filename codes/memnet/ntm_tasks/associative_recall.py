import math
import numpy as np


class AssociativeRecall(object):
    """
        Inputs and targets are binary vectors. There are 2 types of items,
        input item (I) and the query item (Q),
        Both I and Q has length of M (in the paper M=3),
        The number of features that I and Q has is 6, there are 2 additional
        delimiters that makes the number of input dimensions to 8.

        The input X and output Y are represented as a 3d tensor of dimensions:
            t: # timesteps
            mb: size of the minibatch
            f: # of features

        The size would be:
            t x mb x f

        According to the paper, the size of the input is 8 and there are 2
        additional dimensions for the delimiters in the input feature space.
        The first 8 features in the input space are corresponding to the
        input-space features and the last 2 are reserved for the delimiters.

        For example if the beginning of sequence delimiter would be:
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        and the end of sequence delimiter would be:
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    def __init__(self,
                 batch_size=128,
                 max_nitems=6,
                 item_len=3,
                 inp_size=8,
                 rng=None,
                 inc_slope=1e-5,
                 seed=1,
                 use_qmask=False,
                 rand_itemlens=False,
                 n_delimiters=2,
                 rnd_len=False):

        if rng is None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = rng

        assert max_nitems > 2
        self.inp_size = inp_size
        self.rnd_len = rnd_len
        self.seed = seed
        self.batch_size = batch_size
        self.max_nitems = max_nitems
        self.item_len = item_len
        self.inc_slope = inc_slope
        self.rand_itemlens = rand_itemlens
        self.cnt = 0
        self.n_delimiters = n_delimiters
        self.use_qmask = use_qmask
        self.seq_len = (item_len + 1) * (max_nitems + 2)

        if rnd_len:
            self.fn = self.__get_data_rnd
        else:
            self.fn = self.__get_data

    def __iter__(self):
        return self

    def __output_format(self, inp, out, mask=None, cost_mask=None, qmask=None):
        output = {}
        output['x'] = inp
        output['y'] = out

        if mask is not None:
            output['mask'] = mask

        if cost_mask is not None:
            output['cmask'] = cost_mask

        if self.use_qmask and qmask is not None:
            output['qmask'] = qmask

        return output

    def __get_data(self):
        """
            This function will return fixed-length sequences, it should only be used
            for validation and testing.
        """
        input_seq_len = (self.item_len + 1) * (self.max_nitems  + 1) + 1
        sequence = self.rng.binomial(1, 0.5, size=(self.item_len,
                                                   self.max_nitems,
                                                   self.batch_size,
                                                   self.inp_size - self.n_delimiters)).astype(np.float32)
        query_inp = self.rng.random_integers(0, self.max_nitems - 2, size=(self.batch_size,))
        target_inp = query_inp + 1


        input_sequence  = np.zeros((self.item_len + 1,
                                    self.max_nitems + 2,
                                    self.batch_size,
                                    self.inp_size),
                                    dtype=np.float32)

        output_sequence = np.zeros((self.item_len + 1,
                                    self.max_nitems + 2,
                                    self.batch_size,
                                    self.inp_size),
                                    dtype=np.float32)

        cost_mask = np.zeros((self.seq_len, self.batch_size, self.inp_size),
                              dtype=np.float32)

        mask  = np.ones((self.seq_len, self.batch_size), dtype=np.float32)
        input_sequence[1:, :self.max_nitems, :, :-self.n_delimiters]  = sequence
        input_sequence[0, :self.max_nitems, :, -self.n_delimiters] = np.float32(1)
        input_sequence[0, self.max_nitems:self.max_nitems+2, :, -1] = np.float32(1)

        query_els = input_sequence[1:, query_inp, np.arange(self.batch_size), :]

        input_sequence[1:, self.max_nitems, :, :] = query_els
        retrieved_els = input_sequence[1:, target_inp, np.arange(self.batch_size), :]
        output_sequence[1:, self.max_nitems + 1, :, :] = retrieved_els

        input_sequence = input_sequence.transpose(1, 0, 2, 3).reshape((self.seq_len,
                                                 self.batch_size, self.inp_size))
        output_sequence = output_sequence.transpose(1, 0, 2, 3).reshape((self.seq_len,
                                                 self.batch_size, self.inp_size))
        qmask = None
        if self.use_qmask:
            qmask = np.zeros((self.seq_len, self.batch_size), dtype=np.float32)
            qmask[input_seq_len:, :] = np.float32(1)

        cost_mask[input_seq_len:, :, :-self.n_delimiters] = np.float32(1)
        return self.__output_format(input_sequence, output_sequence, mask,
                                    cost_mask, qmask=qmask)

    def __get_data_rnd(self):
        """
            This function will create random length sequences for a minibatch. It will
            return a dictionary formatted by output_format function.
        """
        lower_bound = 2
        upper_bound = self.max_nitems

        if self.inc_slope is not None:
            inc = self.cnt * self.inc_slope
            upper_bound = min(np.floor(self.max_nitems * 0.72 + inc),
                              self.max_nitems)

        #Sample random lengths for the minibatch.
        rand_nitems = self.rng.random_integers(lower_bound,
                                               upper_bound,
                                               size=(self.batch_size,))

        input_sequence  = np.zeros((self.item_len + 1,
                                    self.max_nitems + 2,
                                    self.batch_size,
                                    self.inp_size),
                                    dtype=np.float32)

        output_sequence = np.zeros((self.item_len + 1,
                                    self.max_nitems + 2,
                                    self.batch_size,
                                    self.inp_size),
                                    dtype=np.float32)

        cost_mask = np.zeros((self.seq_len,
                              self.batch_size,
                              self.inp_size),
                              dtype=np.float32)

        qmask = np.zeros((self.seq_len, self.batch_size), dtype=np.float32)
        mask = np.zeros((self.seq_len, self.batch_size), dtype=np.float32)

        for i, rnd_nitem in enumerate(rand_nitems):
            if self.rng.uniform(0, 1) > 0.9:
                rnd_item = self.rng.random_integers(self.max_nitems-2, self.max_nitems)
            if self.rand_itemlens:
                item_len = self.rng.randint(2, self.item_len + 1)
            else:
                item_len = self.item_len

            sequence = self.rng.binomial(1, 0.5, size=(item_len, rnd_nitem,
                                         self.inp_size - self.n_delimiters)).astype(np.uint8)

            query_inp = self.rng.random_integers(0, rnd_nitem - 2)

            target_inp = query_inp + 1
            whole_len = (item_len + 1) * (rnd_nitem + 2)

            input_sequence[1:item_len + 1, :rnd_nitem, i, :-self.n_delimiters] = sequence
            input_sequence[0, :rnd_nitem, i, -self.n_delimiters] = np.float32(1)
            input_sequence[0, rnd_nitem:rnd_nitem + 2, i, -1] = np.float32(1)

            query_el = input_sequence[1:, query_inp, i, :]
            retrieved_el = input_sequence[1:, target_inp, i, :]

            input_sequence[1:, rnd_nitem, i, :] = query_el
            output_sequence[1:, rnd_nitem + 1, i, :] = retrieved_el

            mask[:whole_len, i] = 1
            cost_mask[(whole_len - item_len):whole_len, i, :-self.n_delimiters] = np.float32(1)
            if self.use_qmask:
                qmask[(whole_len - item_len):whole_len, i] = np.float32(1)

        input_sequence = input_sequence.transpose(1, 0, 2, 3).reshape((self.seq_len,
                                                 self.batch_size, self.inp_size))
        output_sequence = output_sequence.transpose(1, 0, 2, 3).reshape((self.seq_len,
                                                 self.batch_size, self.inp_size))
        self.cnt += self.batch_size
        return self.__output_format(input_sequence, output_sequence, mask,
                                    cost_mask, qmask=qmask)

    def next(self):
        return self.fn()

if __name__=="__main__":

    copydatagen = AssociativeRecall(batch_size=8,
                                    use_qmask=True,
                                    rnd_len=True)

    batch = copydatagen.next()
    batch2 = copydatagen.next()

    import ipdb; ipdb.set_trace()

    print batch['x'], batch['x'].shape
    print batch['y'], batch['y'].shape
