import math
import numpy as np


class RepeatCopyDataGen(object):

    def __init__(self,
                 batch_size=128,
                 max_len=100,
                 inp_size=8,
                 rng=None,
                 inc_slope=3*1e-4,
                 seed=1,
                 n_delimiters=2,
                 rnd_len=False,
                 num_repeat = 3,
                 max_repeats = 20,
                 plot_data = False):

        if rng is None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = rng

        assert num_repeat < max_repeats, "Number of repeats should be less than maximum number of repeats."

        self.inp_size = inp_size
        self.rnd_len = rnd_len
        self.seed = seed
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_repeats = max_repeats
        self.inc_slope = inc_slope
        self.cnt = 0
        self.plot_data = plot_data
        self.n_delimiters = n_delimiters
        self.num_repeat = num_repeat
        #Note that it should be like that the normalization shouldn't be minibatch-wise
        self.scalar = np.arange(max_repeats).astype("float32")
        self.scalar = self.scalar / self.scalar.sum()
        means = self.scalar[:self.num_repeat].mean()
        std = self.scalar[:self.num_repeat].std()
        self.scalar = (self.scalar - means) / std

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
        n_repeat = self.num_repeat
        sequence_length = self.max_len
        #This is the total length of input and output
        whole_len = (sequence_length + 1) * (self.num_repeat + 1)
        sequence = self.rng.binomial(1, 0.5, size=(sequence_length + 1, self.batch_size, self.inp_size - self.n_delimiters)).astype(np.float32)
        """
        In the paper there is a space between each item.
        """
        sequence[-1, :, :] = np.cast["float32"](0)
        input_sequence  = np.zeros((whole_len,
                                    self.batch_size,
                                    self.inp_size), dtype=np.float32)

        output_sequence = np.zeros((whole_len,
                                    self.batch_size,
                                    self.inp_size - 1),
                                    dtype=np.float32)

        cost_mask = np.zeros((whole_len,
                                    self.batch_size,
                                    self.inp_size - 1),
                                    dtype=np.float32)

        mask  = np.ones((whole_len,
                         self.batch_size),
                         dtype=np.float32)

        n_repeat_norm = self.scalar[n_repeat - 1]

        repeated_sequence = np.concatenate([sequence for _ in xrange(n_repeat)])

        input_sequence[:sequence_length + 1, :, :-self.n_delimiters]  = sequence
        input_sequence[sequence_length, :, -1] = np.float32(1)
        input_sequence[sequence_length, :, -self.n_delimiters] = n_repeat_norm
        output_sequence[sequence_length + 1:, :, :-1] = repeated_sequence
        output_sequence[-1, :, -self.n_delimiters + 1] = np.float32(1)
        cost_mask[sequence_length + 1:, :, :] = np.float32(1)

        return self.__output_format(input_sequence, output_sequence, mask, cost_mask)

    def __get_data_rnd(self):
        lower_bound = 1
        upper_bound = self.max_len

        if self.inc_slope is not None:
            inc = self.cnt * self.inc_slope
            upper_bound = min(np.floor(2.0 * self.max_len / 3.0 + inc), self.max_len)
            #lower_bound = 2 + min(2.0 + inc, np.ceil(self.max_len * 0.5))

        rand_lens = self.rng.random_integers(lower_bound,
                                             upper_bound,
                                             size=(self.batch_size,))

        n_repeat = self.rng.random_integers(1,
                                             self.num_repeat,
                                             size=(self.batch_size,))

        n_repeat_norm = self.scalar[n_repeat - 1]

        sequence_length = self.max_len
        whole_len = (sequence_length + 1 ) * (self.num_repeat + 1)
        input_sequence  = np.zeros((whole_len,
                                    self.batch_size,
                                    self.inp_size),dtype=np.float32)

        output_sequence = np.zeros((whole_len,
                                    self.batch_size,
                                    self.inp_size-1),dtype=np.float32)

        cost_mask = np.zeros((whole_len,
                              self.batch_size,
                              self.inp_size-1),dtype=np.float32)


        mask  = np.zeros((whole_len,
                          self.batch_size),dtype=np.float32)

        for i, rnd_len in enumerate(rand_lens):
            rnd_seq_len = rnd_len
            sequence = self.rng.binomial(1, 0.5, size=(rnd_seq_len + 1, self.inp_size - self.n_delimiters)).astype(np.uint8)
            sequence[-1,:] = np.cast["float32"](0)
            input_sequence[:rnd_seq_len + 1, i, :-self.n_delimiters] = sequence
            input_sequence[rnd_seq_len, i, -1] = np.float32(1)
            input_sequence[rnd_seq_len, i, -self.n_delimiters] = n_repeat_norm[i]

            repeat = n_repeat[i]
            whole_rnd_len = (rnd_seq_len + 1) * (repeat + 1)
            repeated_sequence = np.concatenate([sequence for _ in xrange(repeat)])
            
            output_sequence[rnd_seq_len + 1 : whole_rnd_len ,i, :-1] = repeated_sequence
            output_sequence[whole_rnd_len -1 ,i, -self.n_delimiters + 1] = np.float32(1)
            cost_mask[rnd_seq_len + 1 : whole_rnd_len, i ,:] = np.float32(1) 
            mask[:whole_rnd_len, i] = np.float32(1)
        self.cnt += self.batch_size
        return self.__output_format(input_sequence, output_sequence, mask, cost_mask)

    def next(self):
        return self.fn()

if __name__=="__main__":
    repeatcopydatagen = RepeatCopyDataGen(batch_size=8,
                                          max_len=10,
                                          rnd_len=True,
                                          inp_size=5,
                                          num_repeat=3,
                                          plot_data = True)
    batch = repeatcopydatagen.next()
    batch2 = repeatcopydatagen.next()
    if repeatcopydatagen.plot_data:
        import matplotlib 
        matplotlib.use("Agg")
        from matplotlib import pyplot
        x = batch['x']
        y = batch['y']
        import numpy
        import itertools
        x = numpy.swapaxes(x,0,1)
        y = numpy.swapaxes(y,0,1)
        nrows = 4
        ncols = 2
        figure, axes = pyplot.subplots(nrows=nrows, ncols=ncols)
        for n, (i, j) in enumerate(itertools.product(xrange(nrows), xrange(ncols))):
            ax = axes[i][j]
            ax.axis('off')
            ax.imshow(numpy.concatenate([x[n].T, y[n].T]), cmap = "gray", interpolation = "none")
        pyplot.show()
        pyplot.savefig('x_y_plot.jpg', dpi = 300)

    import ipdb; ipdb.set_trace()
    print batch['x'], batch['x'].shape
    print batch['y'], batch['y'].shape


