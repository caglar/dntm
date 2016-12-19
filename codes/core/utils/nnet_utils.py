from core.commons import EPS
import theano.tensor as TT


def logsumexp(x, axis=None):
    x_max = TT.max(x, axis=axis, keepdims=True)
    z = TT.log(TT.sum(TT.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def kl_divergence(p, p_hat):
    term1 = p * TT.log(TT.maximum(p, EPS))
    term2 = p * TT.log(TT.maximum(p_hat, EPS))
    term3 = (1 - p) * TT.log(TT.maximum(1 - p, EPS))
    term4 = (1 - p) * TT.log(TT.maximum(1 - p_hat, EPS))
    return term1 - term2 + term3 - term4


def softmax3(z):
    assert z.ndim == 3
    z = z - z.max(-1, keepdims=True)
    out = TT.exp(z) / TT.exp(z).sum(-1, keepdims=True)
    return out


def get_hard_vals(ws, axis=-1, tot_size=120):
    w_idxs = ws.argmax(axis=axis)
    w_1hot = TT.extra_ops.to_one_hot(w_idxs, tot_size)
    return w_1hot
