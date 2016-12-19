import math

import numpy as np


class CopyDataGen(object):
    """
        Inputs and targets are binary vectors. There are 2 delimiters, one is corresponding
        to Beginning of Sequence (BOS) and the other is for End of Sequence (EOS).
        The input X and output Y are represented as a 3-d tensor of dimensions:
            t: # timesteps
            mb: size of the minibatch
            f: # of features

        The size would be:
            t x mb x f

        According to the paper, the size of the input is 8 and there are 2 additional dimensions
        for the delimiters in the input feature space. The first 8 features in the input space
        are corresponding to the input-space features and the last 2 are reserved for the delimiters.

        For example if the beginning of sequence delimiter would be:
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        and the end of sequence delimiter would be:
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        For a sequence of length N, in the input the t=0 would be for BOS delimiter
        t=N+2 would be EOS and in between t=0 and t=N+2, we will have the inputs.

        Note that the total length of input would be 2*N+2, from N+2 to 2*N+2 the input
        will be zero. For the target the opposite would be the true, for 0 to N+2 targets
        are zero and N+2 to the 2*N+2 we will have the targets. Cost-mask is formatted in a
        similar way.
    """
    def __init__(self,
                 batch_size=128,
                 max_len=100,
                 inp_size=8,
                 rng=None,
                 inc_slope=3*1e-4,
                 seed=1,
                 n_delimiters=2,
                 rnd_len=False):

        if rng is None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = rng

        self.inp_size = inp_size
        self.rnd_len = rnd_len
        self.seed = seed
        self.batch_size = batch_size
        self.max_len = max_len
        self.inc_slope = inc_slope
        self.cnt = 0
        self.n_delimiters = n_delimiters

        if rnd_len:
            self.fn = self.__get_data_rnd
        else:
            self.fn = self.__get_data

    def __iter__(self):
        return self

    def __output_format(self, inp, out, mask=None, cost_mask=None):
        output = {}
        output['x'] = inp
        output['y'] = out

        if mask is not None:
            output['mask'] = mask
        if cost_mask is not None:
            output['cmask'] = cost_mask

        return output

    def __get_data(self):
        """
           This function will return fixed-length sequences, it should only be used
           for validation and testing.
        """
        sequence_length = self.max_len
        sequence = self.rng.binomial(1, 0.5, size=(sequence_length, self.batch_size,
                                     self.inp_size - self.n_delimiters)).astype(np.float32)

        input_sequence  = np.zeros((sequence_length * 2 + self.n_delimiters,
                                    self.batch_size,
                                    self.inp_size), dtype=np.float32)

        output_sequence = np.zeros((sequence_length * 2 + self.n_delimiters,
                                    self.batch_size,
                                    self.inp_size),
                                    dtype=np.float32)

        cost_mask = np.zeros((sequence_length * 2 + self.n_delimiters,
                              self.batch_size,
                              self.inp_size),
                              dtype=np.float32)

        mask  = np.ones((sequence_length * 2 + 2,
                         self.batch_size),
                         dtype=np.float32)

        input_sequence[1:sequence_length + 1, :, :-self.n_delimiters]  = sequence
        input_sequence[sequence_length + 1, :, -self.n_delimiters + 1] = np.float32(1)
        input_sequence[0, :, -self.n_delimiters] = np.float32(1)
        output_sequence[sequence_length + self.n_delimiters:, :, :-self.n_delimiters] = sequence
        cost_mask[sequence_length + self.n_delimiters:, :, :-self.n_delimiters] = np.float32(1)
        return self.__output_format(input_sequence, output_sequence, mask, cost_mask)

    def __get_data_rnd(self):
        """
            This function will create random length sequences for a minibatch. It will
            return a dictionary formatted by output_format function.
        """
        lower_bound = 1
        upper_bound = self.max_len

        if self.inc_slope is not None:
            inc = self.cnt * self.inc_slope
            upper_bound = min(np.floor(self.max_len*0.5 + inc), self.max_len)

        #Sample random lengths for the minibatch.
        rand_lens = self.rng.random_integers(lower_bound,
                                             upper_bound,
                                             size=(self.batch_size,))

        sequence_length = self.max_len

        input_sequence  = np.zeros((sequence_length * 2 + self.n_delimiters,
                                    self.batch_size,
                                    self.inp_size),dtype=np.float32)

        output_sequence = np.zeros((sequence_length * 2 + self.n_delimiters,
                                    self.batch_size,
                                    self.inp_size), dtype=np.float32)

        cost_mask = np.zeros((sequence_length * 2 + self.n_delimiters,
                              self.batch_size,
                              self.inp_size),dtype=np.float32)

        mask  = np.zeros((sequence_length * 2 + self.n_delimiters,
                          self.batch_size),
                          dtype=np.float32)

        for i, rnd_len in enumerate(rand_lens):
            if self.rng.uniform(0, 1) > 0.9:
                rnd_seq_len = self.rng.random_integers(self.max_len - 2, self.max_len)
            else:
                rnd_seq_len = rnd_len

            whole_len = 2 * rnd_seq_len + self.n_delimiters
            sequence = self.rng.binomial(1, 0.5, size=(rnd_seq_len, self.inp_size - self.n_delimiters)).astype(np.uint8)
            input_sequence[1:rnd_seq_len + 1, i, :-self.n_delimiters] = sequence
            input_sequence[rnd_seq_len + 1, i, -self.n_delimiters + 1] = np.float32(1)
            input_sequence[0, i, -self.n_delimiters] = np.float32(1)
            output_sequence[rnd_seq_len + self.n_delimiters : whole_len, i, :-self.n_delimiters] = sequence
            mask[:whole_len, i] = 1
            cost_mask[rnd_seq_len + self.n_delimiters : whole_len, i, :-self.n_delimiters] = np.float32(1)

        self.cnt += self.batch_size
        return self.__output_format(input_sequence, output_sequence, mask, cost_mask)

    def next(self):
        return self.fn()

if __name__=="__main__":
    copydatagen = CopyDataGen(batch_size=8,
                              max_len=10,
                              rnd_len=False,
                              inp_size=5)

    batch = copydatagen.next()
    batch2 = copydatagen.next()

    import ipdb; ipdb.set_trace()
    print batch['x'], batch['x'].shape
    print batch['y'], batch['y'].shape

