import logging
import numpy as np

import theano.tensor as TT

from core.utils import safe_izip, as_floatX, sample_weights_classic,\
        sharedX
from core.layers import Layer

logger = logging.getLogger(__name__)
logger.disabled = False


class Memory(Layer):
    """
    Base Memory class.
    """
    def __init__(self, mem_nel, mem_size, address_size,
                 n_read_heads, n_write_heads, n_reading_steps,
                 name="Mem"):

        self.mem_nel = mem_nel
        self.mem_size = mem_size
        self.address_size = address_size
        self.n_read_heads = n_read_heads
        self.n_write_heads = n_write_heads
        self.n_reading_steps = n_reading_steps
        self.name = name
        super(Memory, self).__init__()

    def write(self, erase_heads, write_weights_samples, contents, M_):
        raise NotImplementedError("base class does not have this method.")

    def read(self, read_heads_w, read_weights_samples, m_t):
        raise NotImplementedError("base class does not have this method.")


class AddressedMemory(Memory):
    """
    Memory class with learnable address vectors.
    """
    def __init__(self,
                 mem_nel,
                 mem_size,
                 address_size,
                 n_read_heads,
                 n_write_heads,
                 n_reading_steps,
                 learn_addresses=True,
                 name="AddressedMemory"):

        super(AddressedMemory, self).__init__(mem_nel=mem_nel, mem_size=mem_size,
                                              address_size=address_size,
                                              n_read_heads=n_read_heads,
                                              n_write_heads=n_write_heads,
                                              n_reading_steps=n_reading_steps)
        self.learn_addresses = learn_addresses
        self.name = name
        self.init_params()

    def init_params(self):
        if self.learn_addresses:
            if self.address_size > 0:
                self.L = sample_weights_classic(self.mem_nel + 1,
                                                self.address_size,
                                                3, 0.15)
                delta = 0.003
                low_bound = 1 - delta * (self.mem_nel + 1)
                if low_bound <= 0.2:
                    low_bound = 0.4
                logger.info("Lower bound is %f" % low_bound)
                self.L *= np.linspace(1.0, low_bound, self.mem_nel +
                        1).astype("float32").reshape((self.mem_nel + 1, 1))
        else:
            if self.address_size > 0:
                self.L = sample_weights_classic(self.mem_nel + 1,
                                                self.address_size,
                                                3, 0.15)
                self.L *= np.linspace(1.0, 0.65, self.mem_nel +
                        1).astype("float32").reshape((self.mem_nel + 1, 1))

        self.merge_params()
        eps = 1e-6 #np.random.uniform(-1e-6, 1e-6)

        if self.address_size > 0:
            if self.learn_addresses:
                self.params[self.pname("Memory")] = self.L
                self.L = self.params[self.pname("Memory")]
            else:
                self.L = sharedX(self.L)

            self.C = TT.alloc(as_floatX(0. + eps),
                              self.mem_nel + 1, self.mem_size)
            self.M = TT.concatenate([self.C, self.L], axis=1)
        else:
            self.C = TT.alloc(as_floatX(0. + eps),
                              self.mem_nel + 1, self.mem_size)
            self.M = self.C

    @property
    def address_vectors(self):
        return self.M[:, self.mem_size:]

    @property
    def content_vectors(self):
        return self.M[:, :-(self.address_size)]

    def write(self, erase_heads, write_weights_samples, write_heads_w, contents, M_):
        if self.n_write_heads > 1:
            mulaccum = lambda x, y: x * y
            sumaccum = lambda x, y: x + y

            erase_mems = reduce(mulaccum, [(1 - erase_head * write_head) for erase_head, \
                                write_head in safe_izip(erase_heads, write_weights_samples)])
            write_mems = reduce(sumaccum, [write_head * content for write_head, content in \
                                            safe_izip(write_weights_samples, contents)])
            write_weights = TT.concatenate([w.dimshuffle('x', 0, 1) for w in \
                                               write_weights_samples])
        else:
            write_weights, erase_weights, content = write_heads_w[0], \
                                                    erase_heads[0], contents[0]
            write_weights_samples = write_weights_samples[0]
            write_weights_samples_ = write_weights_samples.dimshuffle(0, 1, 'x')

            erase_mems = (1 - erase_weights * write_weights_samples_)
            write_mems = write_weights_samples_ * content

        if self.address_size > 0:
            M_mem = M_[:, 1:, :self.mem_size]
            m_t = TT.set_subtensor(M_mem, \
                    M_mem * erase_mems + write_mems)
        else:
            M_mem = M_[:, 1:, :]
            m_t = TT.set_subtensor(M_mem, \
                    M_mem * erase_mems + write_mems)

        return write_weights, write_weights_samples, m_t

    def read(self, read_heads_w, read_weights_samples, m_t):
        if self.n_read_heads > 1:
            read_weights = TT.concatenate([r.dimshuffle('x', 0, 1) \
                    for r in read_heads_w])
            read_weights_samples = TT.concatenate([r.dimshuffle('x', 0, 1) \
                    for r in read_weights_samples])
            mem_read_t = sum((m_t[:, :-1, :] * \
                    read_weights_samples[i].dimshuffle(0, 1, 'x')).sum(1) \
                    for i in xrange(self.n_read_heads))
        else:
            read_weights = read_heads_w[0]
            read_weights_samples = read_weights_samples[0]
            mem_read_t = (m_t[:, :-1, :] * \
                    read_weights_samples.dimshuffle(0, 1, 'x')).sum(1)

        return read_weights, read_weights_samples, mem_read_t

    def fprop(self):
        raise NotImplementedError("Memory does not have an fprop function!")
