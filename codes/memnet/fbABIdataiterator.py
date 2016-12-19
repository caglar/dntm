import numpy as np
import cPickle as pkl
import six


class FBbABIDataIteratorSingleQ(six.Iterator):

    def __init__(self,
                 batch_size=16,
                 rng=None,
                 seed=None,
                 task_path=None,
                 task_file=None,
                 task_id=None,
                 fact_vocab=None,
                 single_task_batch=False,
                 predict_next_bow=False,
                 max_seq_limit=200,
                 use_qmask=False,
                 max_seq_len=0,
                 max_fact_len=0,
                 use_bow_out=True,
                 random_drop_inp=None,
                 n_tasks=20,
                 mode=None,
                 randomize=True):

        self.batch_size = batch_size

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.RandomState(seed)

        self.task_path = task_path
        self.task_file = task_file
        self.mode = mode
        self.fact_vocab = fact_vocab
        self.task_id = task_id
        self.randomize = randomize
        self.n_tasks = n_tasks
        self.rnd_int = self.rng.random_integers
        self.random_drop_inp = random_drop_inp
        self.single_task_batch = single_task_batch
        self.use_qmask = use_qmask
        self.max_seq_limit = max_seq_limit
        self.use_bow_out = use_bow_out
        self.predict_next_bow = predict_next_bow

        # Keys
        self.task_k = "task_%d"
        self.qa_k = "qa_%d"
        self.set_k = "set_%d"
        self.q_k = "q_%d"
        self.fact_k = "fact_%d"
        self.ans_k = "ans_%d"

        self.reset()
        self.__load_data()

        if self.task_id:
            self.task_id -= 1

        if max_seq_len and max_fact_len:
           self.max_seq_len = max_seq_len
           self.max_fact_len = max_fact_len
        else:
           self.max_seq_len = 0
           self.max_fact_len = 0
           self.__find_max_seq_len()

    def reset(self):
        self.task_offset = 0
        self.qa_offset = 0
        self.set_offset = 0
        self.fact_offset = 0

    def __find_max_seq_len(self):
        max_fact_len = 0
        i = 0
        for task_key in self.task_data:
            if self.task_id is not None:
                task_key = self.task_k % self.task_id
                if i > self.task_id:
                    break

            for qa_key in self.task_data[task_key]:
                seq_len = 0
                for set_key in self.task_data[task_key][qa_key]:
                    for el_key in self.task_data[task_key][qa_key][set_key]:
                        if "fact_" in el_key:
                            seq_len += 1

                        cseq_len = len(self.task_data[task_key][qa_key][set_key][el_key])
                        if cseq_len > max_fact_len:
                            max_fact_len = cseq_len

            if seq_len + 4 > self.max_seq_len:
                self.max_seq_len = seq_len + 4

            i += 1
        self.max_fact_len = max_fact_len

    def __get_fact_len(self, qa):
        max_fact_len = 0
        for set_key in qa:
            for el_key in qa[set_key]:
                if 'fact_' in el_key:
                    max_fact_len += 1
        return max_fact_len

    def __load_data(self):
        self.vocab = pkl.load(open(self.task_path + self.fact_vocab, "rb"))
        self.task_data = pkl.load(open(self.task_path + self.task_file, "rb"))
        self.max_n_tasks = len(self.task_data.keys())
        self.nwords = len(self.vocab.keys())

        assert self.max_n_tasks >= self.n_tasks
        self.task_lens = {k: len(self.task_data[k].keys()) for k in self.task_data.keys()}
        self.n_tasks = len(self.task_data.keys())

    def __get_data(self, task_id, qa_id):
        task_key = self.task_k % task_id
        qa_key = self.qa_k % qa_id
        qa = self.task_data[task_key][qa_key]

    def __get_bow_out(self, facts):
        bow_outs = np.zeros((self.max_seq_len, self.batch_size, self.nwords))
        if self.predict_next_bow:
            fslen = facts.shape[1] - 1
        else:
            fslen = facts.shape[1]

        for j in xrange(facts.shape[2]):
            for i in xrange(fslen):
                if self.predict_next_bow:
                    fidx = i+1
                else:
                    fidx = i
                bow_outs[i, j, facts[:, fidx, j]] = 1
                bow_outs[i, j, 0] = 0
        return bow_outs.astype("float32")

    def __output_format(self, X, y, mask, cmask, qmask, task_ids):
        output = {}

        # Truncate the unnecessary part of the sequence.
        max_seq_len = np.max(cmask.nonzero()[0]) + 1

        output["x"] = X[:, :max_seq_len, :].astype("int32")
        output["y"] = y[:max_seq_len, :].astype("int32")
        output["mask"] = mask[:max_seq_len, :].astype("float32")
        output["cmask"] = cmask[:max_seq_len, :].astype("float32")
        output["qmask"] = qmask[:max_seq_len, :].astype("float32")
        output["task_ids"] = task_ids.astype("int32")

        if self.use_bow_out:
            output['bow_out'] = self.__get_bow_out(X)[:max_seq_len, :, :]

        return output

    def __get_empty_buffers(self):
        X = np.zeros((self.max_fact_len, self.max_seq_len, self.batch_size)).astype("int32")
        y = np.zeros((self.max_seq_len, self.batch_size)).astype("int32")
        mask = np.zeros((self.max_seq_len, self.batch_size)).astype("float32")
        cmask = np.zeros((self.max_seq_len, self.batch_size)).astype("float32")
        qmask = np.zeros((self.max_seq_len, self.batch_size)).astype("float32")
        task_ids = np.zeros((self.batch_size)).astype("float32")
        return X, y, mask, cmask, qmask, task_ids

    def __get_nquestions(self, set_dict):
        nq = 0
        for k, v in set_dict.iteritems():
            if "q_" in k:
                nq = int(k[2:])
        return nq

    def __random_batch(self):
        # TODO: simplify the code by using list for the data parts
        X, y, mask, cmask, qmask, _ = self.__get_empty_buffers()

        task_ids = self.rnd_int(low=0,
                                high=self.n_tasks - 1,
                                size=self.batch_size)

        for i in xrange(self.batch_size):
            if self.task_id is None:
                task_key = self.task_k % task_ids[i]
            else:
                task_key = self.task_k % self.task_id
                task_ids[i] = self.task_id

            qa_len = self.task_lens[task_key]
            qa_id = self.rnd_int(low=0, high=qa_len-1, size=1)
            qa_key = self.qa_k % qa_id
            qa = self.task_data[task_key][qa_key]

            if self.max_seq_limit and self.max_seq_limit > 0:
                while self.__get_fact_len(qa) + 2 >= self.max_seq_limit:
                    qa_id = self.rnd_int(low=0, high=qa_len-1, size=1)
                    qa_key = self.qa_k % qa_id
                    qa = self.task_data[task_key][qa_key]

            n_sets = len(qa.items())
            rset_id = self.rnd_int(0, n_sets-1)
            set_key = self.set_k % rset_id
            nqs = self.__get_nquestions(qa[set_key])
            rqn = self.rnd_int(low=0, high=nqs, size=1)

            nfacts = 0
            q_met = False

            for j, (k, setv) in enumerate(qa.iteritems()):
                nas = 1
                for n, (nk, nfactv) in enumerate(setv.iteritems()):
                    flen = len(nfactv)
                    if "ans_" in nk and q_met:
                        y[nfacts + nas, i] = nfactv[0]
                        mask[nfacts + nas, i] = 1
                        cmask[nfacts + nas, i] = 1
                        qmask[nfacts + nas - 1, i] = 1
                        nas += 1
                    else:
                        if "q_" in nk and rqn[0] == int(nk[2:]):
                            X[:flen, nfacts, i] = np.asarray(nfactv).astype("int32")
                            mask[:nfacts + 1, i] = 1
                            q_met = True
                        elif "q_" in nk:
                            if q_met:
                                q_met = False
                                nas = 1
                                break
                        elif "fact_" in nk:
                            if q_met:
                                q_met = False
                                nas = 1
                                break
                            X[:flen, nfacts, i] = np.asarray(nfactv).astype("int32")
                            mask[nfacts, i] = 1
                            nfacts += 1
                else:
                    continue
                break
        return self.__output_format(X, y, mask, cmask, qmask, task_ids)

    def __check_set_offset(self, data):
        set_len = len(data.keys())
        if self.set_offset >= set_len:
            self.set_offset = 0
            self.qa_offset += 1
            task_key = self.task_k % self.task_offset
            self.__check_qa_offset(self.task_data[task_key])
            return True
        else:
            self.set_offset += 1
        return False

    def __inc_batch_idx(self, brange):
        try:
            i = next(bs_range)
        except:
            return False
        return bs_range

    def __check_task_offset(self, data):
        task_len = self.n_tasks
        if self.task_offset >= task_len:
            self.reset()
            if self.mode != "train":
                raise StopIteration
        return False

    def __check_qa_offset(self, data):
        qa_len = len(data.keys())
        if self.qa_offset >= qa_len:
            self.qa_offset = 0
            self.set_offset = 0
            task_key = self.task_k % self.task_offset
            self.__check_task_offset(self.task_data)
            self.task_offset += 1
            return True
        return False

    def __batch(self):
        #TODO: simplify the code by using list for the data parts
        task_len = self.n_tasks

        if self.task_offset >= task_len:
            self.reset()
            if self.mode != 'train':
                raise StopIteration

        X, y, mask, cmask, qmask, task_ids = self.__get_empty_buffers()
        if self.task_id is None:
            task_key = self.task_k % self.task_offset
        else:
            task_key = self.task_k % self.task_id

        qa_len = self.task_lens[task_key]
        nas = 1

        bs_range = iter(xrange(self.batch_size))
        for i in bs_range:
            if self.task_id is not None or self.task_id == 0:
                if self.qa_offset >= qa_len:
                    self.reset()
                    if self.mode != 'train':
                        raise StopIteration

            if self.task_id is None:
                task_key = self.task_k % self.task_offset
                task_ids[i] = self.task_offset
            else:
                task_key = self.task_k % self.task_id
                task_ids[i] = self.task_id

            task_len = self.n_tasks
            qa_len = self.task_lens[task_key]
            qa_key = self.qa_k % self.qa_offset
            qa = self.task_data[task_key][qa_key]

            nfacts = 0
            #
            sets_len = len(qa.items())
            example = np.zeros((self.max_fact_len, self.max_seq_len))

            if self.set_offset >= sets_len:
                self.set_offset = 0
                self.qa_offset += 1
                if self.qa_offset >= qa_len:
                    if self.task_id is not None or self.task_id == 0:
                        self.reset()
                        if self.mode != 'train':
                            raise StopIteration
                    elif self.task_id is None:
                        self.task_offset += 1
                        self.qa_offset = 0
                        if self.task_offset >= task_len:
                            self.reset()
                            if self.mode != 'train':
                                raise StopIteration
                        if self.task_id is None:
                            task_key = self.task_k % self.task_offset
                            task_ids[i] = self.task_offset
                        else:
                            task_key = self.task_k % self.task_id
                            task_ids[i] = self.task_id
                        qa_len = self.task_lens[task_key]

                qa_key = self.qa_k % self.qa_offset
                qa = self.task_data[task_key][qa_key]
                nfacts = 0
                sets_len = len(qa.items())

            for j in xrange(self.set_offset, sets_len):
                set_key = self.set_k % j
                setv = self.task_data[task_key][qa_key][set_key]
                facts_len = len(setv.items())
                for n in xrange(self.fact_offset, facts_len):
                    nk = setv.keys()[n]
                    nfactv = setv[nk]
                    flen = len(nfactv)
                    if "ans_" in nk:
                        y[nfacts + nas, i] = nfactv[0]
                        mask[nfacts + nas, i] = 1
                        cmask[nfacts + nas, i] = 1
                        qmask[nfacts + nas - 1, i] = 1
                        nas += 1
                    else:
                        if "q_" in nk:
                            if nas > 1:
                                nas = 1
                                try:
                                    i = next(bs_range)
                                except:
                                    break
                            else:
                                X[:, :, i] = example
                            mask[:nfacts + 1, i] = 1
                            X[:, :, i] = example
                            X[:flen, nfacts, i] = np.asarray(nfactv).astype("int32")
                        else:
                            if nas > 1:
                                nas = 1
                                example[:flen, nfacts] = np.asarray(nfactv).astype("int32")
                                X[:, :, i] = example
                                mask[:nfacts + 1, i] = 1
                                try:
                                    i = next(bs_range)
                                except:
                                    break
                            else:
                                example[:flen, nfacts] = np.asarray(nfactv).astype("int32")
                            mask[:nfacts+1, i] = 1
                            example[:flen, nfacts] = np.asarray(nfactv).astype("int32")
                            nfacts += 1
                else:
                    if nas > 1:
                        nas = 1
                        self.fact_offset = 0
                        self.set_offset += 1
                        if self.set_offset >= sets_len:
                           break
                        else:
                            try:
                                i = next(bs_range)
                            except:
                                break
                    continue
                break
            else:
                self.set_offset = 0
                self.qa_offset += 1
                qa_len = self.task_lens[task_key]
                if self.qa_offset >= qa_len:
                    self.task_offset += 1
                    if self.task_offset >= task_len:
                        break
                    self.qa_offset = 0
                    self.set_offset = 0
                    self.fact_offset = 0

                if self.single_task_batch:
                    break
                continue

        if not self.__check_task_offset(self.task_data):
            if not self.__check_qa_offset(self.task_data[task_key]):
                qa_key = self.qa_k % self.qa_offset
                self.__check_set_offset(self.task_data[task_key][qa_key])
        return self.__output_format(X, y, mask, cmask, qmask, task_ids)

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == 'train':
            if self.randomize:
                return self.__random_batch()
            else:
                return self.__batch()
        else:
            if self.randomize:
                return self.__random_batch()
            else:
                return self.__batch()

    def __get_state__(self):
        state = {}
        state['task_path'] = self.task_path
        state['task_file'] = self.task_file
        state['fact_vocab'] = self.fact_vocab
        state['randomize'] = self.randomize
        state['n_tasks'] = self.tasks
        state['rng'] = self.rng
        state['use_qmask'] = self.use_qmask
        state['use_bow_out'] = self.use_bow_out
        state['max_seq_limit'] = self.max_seq_limit
        state['predict_next_bow'] = self.predict_next_bow
        state['task_k'] = self.task_k
        state['qa_k'] = self.qa_k
        state['set_k'] = self.set_k
        state['q_k'] = self.q_k
        state['fact_k'] = self.fact_k
        state['ans_k'] = self.ans_k
        state['task_id'] = self.task_id
        state['max_seq_len'] = self.max_seq_len
        state['max_fact_len'] = self.max_fact_len
        state['task_offset'] = self.task_offset
        state['qa_offset'] = self.qa_offset
        state['set_offset'] = self.set_offset
        state['fact_offset'] = self.fact_offset

    def __set_state__(self, state):
        self.__dict__.update(state)
