#!/bin/bash -x

python -m ipdb extract_word2vec_embs.py --vocab /data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en-10k/all_tasks_test_ngram_False_dict.pkl --w2v-dir /data/lisatmp3/chokyun/dictionary/D_cbow_pdw_8B.pkl --emb-size 160 --nwords-pca 90000 --save-dir ./new_dict_ngram_false_all_tasks_160.pkl
