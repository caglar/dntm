import numpy
import ipdb

from StringIO import StringIO
import tokenize
from sklearn.feature_extraction.text import CountVectorizer

import math

import numpy as np
import cPickle

import glob
list_of_files = glob.glob('/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en/*.tok')
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_words(task_file):
    facts = []
    for line in open(task_file,'r'):
        tokens = tokenize.generate_tokens(StringIO(line).readline)
        for x in tokens:
            if is_number(x[1]):
                continue
            facts.append(x[1].lower())
    return facts

tokens = []
for file in list_of_files:
    tokens.extend(read_words(file))


fact_dict = '/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en/global_fact_dict.pkl'
fact_vect = CountVectorizer(min_df=0, token_pattern=r".*")
fact_vect.fit_transform(tokens)
ipdb.set_trace()
f = open(fact_dict, 'w')
cPickle.dump(fact_vect, f)
f.close()
