import numpy as np

import logging
from collections import defaultdict

import cPickle as pkl
from core.basic import MainLoop

import theano

from facebook_bABI_data_gen_2 import facebookBABIDataGen
logger = logging.getLogger(__name__)
logger.disabled = False

class FBaBIMainLoop(MainLoop):

    def __init__(self, model,
                 learning_rate=1e-3,
                 print_every=None,
                 checkpoint_every=None,
                 inspect_every=None,
                 validate_every=None,
                 train_data_gen=None,
                 valid_data_gen=None,
                 report_task_errors=True,
                 reload_model=False,
                 inspect_only=False,
                 valid_iters=None,
                 linear_start=False,
                 use_qmask=False,
                 max_iters=None,
                 prefix=None):

        assert prefix is not None, "Prefix should not be empty."
        logger.info("Building the computational graph.")

        self.model = model
        self.learning_rate = learning_rate
        self.print_every = print_every
        self.train_data_gen = train_data_gen
        self.valid_data_gen = valid_data_gen
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every
        self.valid_iters = valid_iters
        self.inspect_every = inspect_every
        self.inspect_only = inspect_only
        self.reload_model = reload_model
        self.prefix = prefix
        self.report_task_errors = report_task_errors
        self.linear_start = linear_start

        self.patience = 2
        self.inc_ratio = 0.1
        self.prev_val_cost = np.inf
        self.cur_patience = 0

        self.use_qmask = use_qmask

        self.mdl_name = self.prefix + "_model_params.pkl"
        self.stats_name = self.prefix + "_stats.pkl"
        self.best_mdl_name = self.prefix + "_best_model_params.pkl"
        self.best_stats_name = self.prefix + "_best_stats.pkl"

        self.best_cost = np.inf

        self.stats = defaultdict(list)

        self.max_iters = max_iters
        self.train_fn = None
        self.valid_fn = None
        self.cnt = 0
        self.prepare_model()

    def prepare_model(self):
        if self.reload_model:
            logger.info("Reloading stats from %s." % self.mdl_name)
            if not self.inspect_model:
                self.stats = pkl.load(open(self.stats_name, "rb"))
        mdl_name = self.mdl_name if self.reload_model else None

        if not self.inspect_only:
            self.train_fn = self.model.get_train_fn(lr=self.learning_rate,
                                                    mdl_name=mdl_name)

        if self.validate_every and not self.inspect_only:
            self.valid_fn = self.model.get_valid_fn(mdl_name=mdl_name)

        if self.inspect_every or self.inspect_only:
            self.inspect_fn = self.model.get_inspect_fn(mdl_name=mdl_name)

    def train(self):
        batch = self.train_data_gen.next()
        tdata_x = batch['x']
        tdata_y = batch['y']
        tcmask = batch['cmask']
        tmask = batch['mask']

        if self.use_qmask:
            tqmask = batch['qmask']
            cost, gnorm, norm_up, param_norm, errors = self.train_fn(tdata_x, tdata_y,
                                                                     tmask, tcmask,
                                                                     tqmask)
        else:
            cost, gnorm, norm_up, param_norm, errors = self.train_fn(tdata_x, tdata_y,
                                                                     tmask, tcmask)

        if self.cnt % self.print_every == 0:
            self.stats['train_cnt'].append(self.cnt)
            self.stats['train_cost'].append(cost)
            self.stats['norm_up'].append(norm_up)
            self.stats['param_norm'].append(param_norm)
            self.stats['gnorm'].append(gnorm)
            self.stats['errors'].append(errors)

            train_str = ("Iter %d: cost: %f, update norm: %f, "
                         " parameter norm: %f, "
                         " norm of gradients: %f"
                         " errors: %f")

            train_str_vals = (self.cnt, cost, norm_up, param_norm, gnorm, errors)
            logger.info(train_str % train_str_vals)

    def inspect_model(self, data_x, mask, cmask=None, qmask=None):
        logger.info("Inspecting the model.")
        if self.model.tie_read_write_gates:
            h_t, m_t, mem_read_t, write_weights, read_weights, probs = \
                        self.inspect_fn(data_x, mask, cmask)
            probs = probs.reshape(cmask.shape) * cmask
            vals = {"h_t": h_t, "m_t": m_t, "mem_read": mem_read_t,
                    "read_weights": read_weights,
                    "probs": cmask, "data_x": data_x, "mask": mask}
        else:
            h_t, m_t, mem_read_t, write_weights, read_weights, probs = \
                        self.inspect_fn(data_x, mask, cmask)
            probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                    cmask.reshape((cmask.shape[0], cmask.shape[1], 1))
            vals = {"h_t": h_t, "m_t": m_t, "mem_read": mem_read_t,
                    "read_weights": read_weights,
                    "write_weights": write_weights,
                    "probs": probs, "data_x": data_x, "mask": mask}

            if qmask is not None:
                vals["qmask"] = qmask

            if cmask is not None:
                vals['cmask'] = cmask

        inspect_file = self.prefix + "_inspections.pkl"
        pkl.dump(vals, open(inspect_file, "wb"), 2)
    '''
    def validate(self):
        logger.info("Validating the model.")
        logger.info("The norm of the parameters are : ")
        self.model.params.print_param_norms()
        import glob
        l = glob.glob('/data/lisatmp3/gulcehrc/data/tasks_1-20_v1-2/en/qa*test.txt.tok')
        for f in l:
            fname = f.split('/')[-1].split('_test')[0]
            fbBABIdatagen = facebookBABIDataGen(task_filename = fname, task = 'test', batch_size=2)
            batch = fbBABIdatagen.next()
            #batch = self.valid_data_gen.next()
            vdata_x, vdata_y, cmask, mask = batch['x'], batch['y'], batch['cmask'], batch['mask']

            if self.model.softmax:
                cost, errors = self.valid_fn(vdata_x, vdata_y, mask, cmask)
                valid_str_errors = "Valid errors on fname: %f costs: %f"
                valid_str_errors_vals = (errors, cost)
                self.stats['valid_errors'].append(errors)
                logger.info(valid_str_errors % valid_str_errors_vals)
            else:
                cost = self.valid_fn(vdata_x, vdata_y, mask, cmask)
                valid_str_cost = "Valid cost: %f"
                valid_str_cost_vals = (cost)
                logger.info(valid_str % valid_str_cost_vals)

            self.stats['valid_cost'].append(cost)

            if cost <= self.best_cost:
                logger.info("Saving the best model.")
                self.best_cost = cost
                self.save(mdl_name=self.best_mdl_name,
                          stats_name=self.best_stats_name)
    '''
    def validate(self):
        logger.info("Validating the model.")
        costs = []
        errors = []
        if self.valid_iters:
            for i in xrange(self.valid_iters):

                if self.use_qmask:
                    try:
                        batch = self.valid_data_gen.next()
                        vdata_x, vdata_y, cmask, mask, qmask = batch['x'], batch['y'], batch['cmask'], batch['mask'], batch['qmask']
                    except:
                        break
                    cost, error = self.valid_fn(vdata_x, vdata_y, mask, cmask, qmask)
                else:
                    try:
                        batch = self.valid_data_gen.next()
                        vdata_x, vdata_y, cmask, mask = batch['x'], batch['y'], batch['cmask'], batch['mask']
                    except:
                        break
                    cost, error = self.valid_fn(vdata_x, vdata_y, mask, cmask)
                costs.append(cost)
                errors.append(error)
        else:
            taskerrs = defaultdict(list)
            for batch in self.valid_data_gen:
                if self.use_qmask:
                    vdata_x, vdata_y, cmask, mask, qmask, task_ids = batch['x'], batch['y'], batch['cmask'], batch['mask'], batch['qmask'], batch['task_ids']
                    cost, error = self.valid_fn(vdata_x, vdata_y, mask, cmask, qmask)
                    taskerrs[task_ids[0][0]].append(error)
                else:
                    vdata_x, vdata_y, cmask, mask, task_ids = batch['x'], batch['y'], batch['cmask'], batch['mask'], batch['task_ids']
                    cost, error = self.valid_fn(vdata_x, vdata_y, mask, cmask)
                    taskerrs[task_ids[0][0]].append(error)

                costs.append(cost)
                errors.append(error)

            if self.report_task_errors:
                for k, v in taskerrs.iteritems():
                    print "Task %d, error is: %f." % (k, np.mean(v))

        error = np.mean(errors)
        cost = np.mean(costs)

        valid_str_errors = "Valid errors: %f costs: %f"
        valid_str_errors_vals = (error, cost)
        self.stats['valid_errors'].append(error)

        logger.info(valid_str_errors % valid_str_errors_vals)

        self.stats['valid_cost'].append(cost)
        logger.info("The norm of the parameters are : ")
        self.model.params.print_param_norms()

        if cost <= self.best_cost:
            logger.info("Saving the best model.")
            self.best_cost = cost
            self.save(mdl_name=self.best_mdl_name,
                      stats_name=self.best_stats_name)

        if self.linear_start:
            if error - self.prev_error > self.inc_ratio:
                if self.cur_patience > self.patience:
                    self.model.out_layer.init_params()
                    self.model.add_noise_to_params()
                    self.cur_patience -= 1
            else:
                self.cur_patience += 1

            self.prev_error = error

    def save(self, mdl_name=None, stats_name=None):
        if mdl_name is None:
            mdl_name = self.mdl_name

        if stats_name is None:
            stats_name = self.stats_name

        logger.info("Saving the model to %s." % mdl_name)
        self.model.params.save(mdl_name)
        pkl.dump(self.stats, open(stats_name, 'wb'), 2)

    def run(self):
        running = True
        logger.info("Started running...")
        while running:
            if self.validate_every:
                if self.cnt % self.validate_every == 0:
                    self.validate()

            self.train()
            if (self.inspect_every is not None
                    and self.cnt % self.inspect_every != 0):
                self.inspect_model()

            if self.cnt % self.checkpoint_every == 0:
                self.save()

            if self.cnt >= self.max_iters:
                running = False

            self.cnt += 1


