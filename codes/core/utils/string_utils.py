""" Utilities for modifying strings"""

import os
import re

from theano.compat.six import string_types
from theano.compat.six.moves import xrange

def preprocess(string, environ=None):
    """
    Preprocesses a string, by replacing `${VARNAME}` with
    `os.environ['VARNAME']` and ~ with the path to the user's
    home directory

    Parameters
    ----------
    string : str
        String object to preprocess
    environ : dict, optional
        If supplied, preferentially accept values from
        this dictionary as well as `os.environ`. That is,
        if a key appears in both, this dictionary takes
        precedence.

    Returns
    -------
    rval : str
        The preprocessed string
    """

    if environ is None:
        environ = {}

    split = string.split('${')

    rval = [split[0]]

    for candidate in split[1:]:
        subsplit = candidate.split('}')

        if len(subsplit) < 2:
            raise ValueError('Open ${ not followed by } before '
                             'end of string or next ${ in "' + string + '"')

        varname = subsplit[0]
        try:
            val = (environ[varname] if varname in environ
                   else os.environ[varname])
        except KeyError:
           raise ValueError('Unrecognized environment variable "' +
                                  varname + '". Did you mean ' +
                                  match(varname, os.environ.keys()) + '?')

        rval.append(val)

        rval.append('}'.join(subsplit[1:]))

    rval = ''.join(rval)

    string = os.path.expanduser(string)

    return rval


def find_number(s):
    """
    Returns None if there are no numbers in the string. Otherwise,
    returns the range of characters occupied by the first number in
    the string.

    Parameters
    ----------
    s : str
        The string to search

    Returns
    -------
    output : tuple or None
        None if no number found
        Tuple containing the range of character indices of the first
        number found otherwise.
    """

    r = re.search('[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?', s)
    if r is not None:
        return r.span(0)
    return None


def tokenize_by_number(s):
    """
    Splits a string into a list of tokens. Each is either a string
    containing no numbers or a float.

    Parameters
    ----------
    s : str
        The string to be tokenized.

    Returns
    -------
    output : list
        A list. Each element is either a float, or a string containing
        no numbers.
    """

    r = find_number(s)

    if r is None:
        return [s]
    else:
        tokens = []
        if r[0] > 0:
            tokens.append(s[0:r[0]])
        tokens.append(float(s[r[0]:r[1]]))
        if r[1] < len(s):
            tokens.extend(tokenize_by_number(s[r[1]:]))
        return tokens
    assert False  # line should be unreached


def match(wrong, candidates):
    """
    Returns a guess of which candidate is the right one
    based on the wrong word.

    Parameters
    ----------
    wrong : str
        A mispelling
    candidates : list of str
        A set of correct words

    Returns
    -------
    WRITEME

    Notes
    -----
    This should be used with a small number of candidates and a high
    potential edit distance (i.e. use it to correct a wrong filename in
    a directory, wrong class name in a module, etc.) Don't use it to
    correct small typos of freeform natural language words.
    """

    assert len(candidates) > 0

    # Current implementation tries all candidates and outputs the one
    # with the min score
    # Could try to do something smarter

    def score(w1, w2):
        # Current implementation returns negative dot product of
        # the two words mapped into a feature space by mapping phi
        # w -> [ phi(w1), .1 phi(first letter of w), .1 phi(last letter of w) ]
        # Could try to do something smarter

        w1 = w1.lower()
        w2 = w2.lower()

        def phi(w):
            # Current feature mapping is to the vector of counts of
            # all letters and two-letter sequences
            # Could try to do something smarter
            rval = {}

            for i in xrange(len(w)):
                l = w[i]
                rval[l] = rval.get(l, 0.) + 1.
                if i < len(w) - 1:
                    b = w[i:i + 2]
                    rval[b] = rval.get(b, 0.) + 1.

            return rval

        def mul(d1, d2):
            rval = 0

            for key in set(d1).union(d2):
                rval += d1.get(key, 0) * d2.get(key, 0)

            return rval

        tot_score = mul(phi(w1), phi(w2)) / float(len(w1) * len(w2)) + \
            0.1 * mul(phi(w1[0:1]), phi(w2[0:1])) + \
            0.1 * mul(phi(w1[-1:]), phi(w2[-1:]))

        return tot_score

    scored_candidates = [(-score(wrong, candidate), candidate)
                         for candidate in candidates]

    scored_candidates.sort()

    return scored_candidates[0][1]


def censor_non_alphanum(s):
    """
    Returns s with all non-alphanumeric characters replaced with *

    Parameters
    ----------
    s : str
        The string to be censored.

    Returns
    -------
    output : str
        The censored version of `s`
    """

    def censor(ch):
        if (ch >= 'A' and ch <= 'z') or (ch >= '0' and ch <= '9'):
            return ch
        return '*'

    return ''.join(censor(ch) for ch in s)


