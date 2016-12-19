import os
import logging
from collections import defaultdict, OrderedDict

import numpy as np

import cPickle as pkl
import theano

from core.basic import MainLoop
from core.commons import SAVE_DUMP_FOLDER
from core.utils import ensure_dir_exists

logger = logging.getLogger(__name__)
logger.disabled = False


class FBaBIMainLoop(MainLoop):
    """
        MainLoop for the Facebook bAbI tasks.
    """
    def __init__(self, model,
                 learning_rate=1e-3,
                 print_every=None,
                 checkpoint_every=None,
                 inspect_every=None,
                 validate_every=None,
                 train_data_gen=None,
                 valid_data_gen=None,
                 test_data_gen=None,
                 train_mon_data_gen=None,
                 report_task_errors=True,
                 reload_model=False,
                 inspect_only=False,
                 valid_only=False,
                 inspect_model=None,
                 valid_iters=None,
                 monitor_grad_norms=True,
                 memory_net_training=False,
                 monitor_full_train=False,
                 bow_weight_start=1.0,
                 bow_weight_stop=0.1,
                 bow_weight_anneal_start=500,
                 bow_out_anneal_rate=4e-5,
                 dice_inc=0.0015,
                 linear_start=False,
                 use_qmask=False,
                 max_iters=None,
                 state=None,
                 prefix=None):

        assert prefix is not None, "Prefix should not be empty."
        logger.info("Building the computational graph.")

        super(FBaBIMainLoop, self).__init__(model=model,
                                            learning_rate=learning_rate,
                                            checkpoint_every=checkpoint_every,
                                            print_every=print_every,
                                            inspect_every=inspect_every,
                                            inspect_only=inspect_only,
                                            valid_only=valid_only,
                                            monitor_full_train=monitor_full_train,
                                            train_data_gen=train_data_gen,
                                            train_mon_data_gen=train_mon_data_gen,
                                            test_data_gen=test_data_gen,
                                            max_iters=max_iters,
                                            valid_data_gen=valid_data_gen,
                                            validate_every=validate_every,
                                            reload_model=reload_model,
                                            prefix=prefix)

        self.pred_bow_out = model.predict_bow_out
        self.dice_inc = dice_inc
        self.bow_weight_start = bow_weight_start
        self.bow_weight_stop = bow_weight_stop
        self.bow_out_anneal_rate = bow_out_anneal_rate
        self.bow_weight_anneal_start = bow_weight_anneal_start

        self.report_task_errors = report_task_errors
        self.linear_start = linear_start
        self.memory_net_training = memory_net_training
        self.monitor_grad_norms = monitor_grad_norms

        self.use_qmask = use_qmask
        self.mdl_name = self.prefix + "_model_params.pkl"
        self.stats_name = self.prefix + "_stats.pkl"
        self.best_mdl_name = self.prefix + "_best_model_params.pkl"
        self.best_stats_name = self.prefix + "_best_stats.pkl"

        self.stats = defaultdict(list)
        self.use_reinforce = self.model.use_reinforce
        self.use_reinforce_baseline = self.model.use_reinforce_baseline
        self.state = state
        self.max_iters = max_iters
        self.train_fn = None
        self.valid_fn = None
        self.cnt = 0
        self.is_best_valid = False
        self.prepare_model()
        self.trainpartitioner = self.model.trainpartitioner
        self.comp_grad_fn = self.model.comp_grad_fn

    def print_train_grad_norms(self):
        logger.info("Printing out the norms")

        for gm in self.trainpartitioner.gs:
            name = gm.name
            norm = ((gm.get_value()**2).sum())**.5
            logger.info("%s : %.3f" % (name, norm))

    def train(self):

        batch = self.train_data_gen.next()

        tdata_x = batch['x']
        tdata_y = batch['y']
        tcmask = batch['cmask']
        tmask = batch['mask']
        xlen = tdata_y.shape[0]

        avg_gnorm = 0.
        avg_cost = 0.
        avg_norm_up = 0.
        avg_errors = 0.
        dice_val = 1. / (1 + self.cnt * self.dice_inc)**0.5

        if self.pred_bow_out:
            if self.cnt >= self.bow_weight_anneal_start:
                boww_diff = abs(
                    self.bow_weight_start - self.bow_weight_stop)
                nsteps = self.cnt - self.bow_weight_anneal_start
                boww_delta = self.bow_out_anneal_rate * nsteps * boww_diff
                self.bow_weight = max(
                    self.bow_weight_stop,
                    self.bow_weight_start - boww_delta)
            elif self.bow_weight_start > 0 and self.bow_weight_anneal_start > 0:
                self.bow_weight = self.bow_weight_start
            else:
                self.bow_weight = 0.0

        if self.memory_net_training:
            tdata_q = batch['q']
            cost, gnorm, norm_up, param_norm, errors = self.train_fn(
                tdata_x, tdata_q, tdata_y, tmask, tcmask, xlen)
        else:
            if self.use_qmask and self.model.correlation_ws:
                tqmask = batch['qmask']
                inps = OrderedDict({'X': tdata_x, 'y': tdata_y, 'mask': tmask,
                                    'cost_mask': tcmask,
                                    'qmask':tqmask})
            else:
                inps = OrderedDict({'X': tdata_x, 'y': tdata_y, 'mask': tmask,
                                    'cost_mask': tcmask})

            if self.pred_bow_out:
                train_bow = batch['bow_out']
                inps.update(OrderedDict({'bow_out': train_bow}))
                inps.update(OrderedDict({"bow_out_w": self.bow_weight}))

            if self.comp_grad_fn:
                self.trainpartitioner.accum_grads(self.comp_grad_fn, inps)

            inps.update({'seq_len': xlen})

            if self.use_reinforce:
                outs = self.train_fn(**inps)
                cost, gnorm, norm_up, param_norm, errors = outs[0], outs[1], \
                        outs[2], outs[3], outs[4]
                idx = 5
                if self.pred_bow_out:
                    bow_cost = outs[idx]
                    idx+=1

                read_const, baseline, read_policy, write_policy = outs[idx], \
                        outs[idx+1], outs[idx+2], outs[idx+3]

                if not self.use_reinforce_baseline:
                     center, cost_std, base_reg = outs[idx+4], outs[idx+5], \
                             outs[idx+6]
            else:
                if self.pred_bow_out:
                    cost, gnorm, norm_up, param_norm, errors, bow_cost = self.train_fn(**inps)
                else:
                    cost, gnorm, norm_up, param_norm, errors = self.train_fn(**inps)

        if self.model.use_dice_val:
            self.model.dice_val.set_value(np.float32(dice_val))

        avg_cost = 0.9 * avg_cost + 0.1 * cost
        avg_gnorm = 0.9 * avg_gnorm +  0.1 * gnorm
        avg_norm_up = 0.9 * avg_norm_up + 0.1 * norm_up
        avg_errors = 0.9 * avg_errors + 0.1 * errors

        if errors > 1.0:
            import ipdb; ipdb.set_trace()

        if self.cnt % self.print_every == 0:
            self.stats['train_cnt'] = self.cnt
            self.stats['train_cost'].append(cost)
            self.stats['norm_up'].append(norm_up)
            self.stats['param_norm'].append(param_norm)
            self.stats['gnorm'].append(gnorm)
            self.stats['errors'].append(errors)

            if self.pred_bow_out:
                self.stats['bow_cost'].append(bow_cost)

            if self.use_reinforce:
                self.stats['read_const'].append(read_const)
                self.stats['baseline'].append(baseline)
                self.stats['read_policy'].append(read_policy)
                self.stats['write_policy'].append(write_policy)

                if self.model.use_reinforce_baseline:
                    train_str = ("Iter %d: cost: %f, update norm: %.4f, "
                                 " parameter norm: %.4f, "
                                 " norm of gradients: %.3f"
                                 " errors: %.2f"
                                 " read constr %f"
                                 " baseline %f"
                                 " read policy %.3f"
                                 " write policy %.3f"
                                 " dice val %.3f")

                    train_str_vals = (self.cnt, cost.mean(), norm_up,
                                      param_norm, gnorm, errors, read_const,
                                      baseline, read_policy, write_policy, dice_val)
                else:
                    self.stats["center"].append(center)
                    self.stats["cost_std"].append(cost_std)
                    self.stats["base_reg"].append(base_reg)

                    train_str = ("Iter %d: cost: %.4f,"
                                 " update norm: %.3f,"
                                 " parameter norm: %.4f,"
                                 " norm of grads: %.3f"
                                 " errors: %.2f"
                                 " read constr %f"
                                 " baseline %f"
                                 " center %.3f"
                                 " cost_std %.3f"
                                 " base_reg %.3f"
                                 " read poli %.3f"
                                 " write poli %.3f"
                                 " dice val %.3f")

                    baseline = baseline.mean()
                    train_str_vals = (self.cnt, cost, norm_up, param_norm,
                                      gnorm, errors, read_const, baseline,
                                      center, cost_std, base_reg, read_policy,
                                      write_policy, dice_val)

                    if self.pred_bow_out:
                        train_str += "bow_cost %.4f"
                        train_str_vals = tuple(list(train_str_vals) + [bow_cost])
            else:
                if self.pred_bow_out:
                    train_str = ("Iter %d: cost: %f, update norm: %.4f, "
                                " parameter norm: %.4f, "
                                " norm of gradients: %.3f"
                                " errors: %.2f"
                                " bow_cost: %.5f"
                                " hints_w: %.2f")

                    train_str_vals = (self.cnt, cost, norm_up,
                                      param_norm, gnorm, errors, bow_cost,
                                      self.bow_weight)
                else:
                    train_str = ("Iter %d: cost: %f, update norm: %.4f, "
                                " parameter norm: %.4f, "
                                " norm of gradients: %.3f"
                                " errors: %.2f")

                    train_str_vals = (self.cnt, cost, norm_up,
                                      param_norm, gnorm, errors)

            logger.info(train_str % train_str_vals)

    def inspect_model(self, data_x=None,
                      mask=None, cmask=None,
                      qmask=None):

        if data_x is None or mask is None:
            self.valid_data_gen.reset()
            batch = self.valid_data_gen.next()
            data_x = batch['x']
            mask = batch['mask']
            cmask = batch['cmask']
            qmask = batch['qmask']

        xlen = mask.shape[0]

        logger.info("Inspecting the model.")
        if not self.model.smoothed_diff_weights:
            if self.model.use_reinforce:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_samples, read_weights_samples, probs = \
                            self.inspect_fn(data_x, mask, cmask, xlen)

                probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                        cmask.reshape((cmask.shape[0], cmask.shape[1], 1))

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_samples": read_weights_samples,
                        "write_weights_samples": write_weights_samples,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
            else:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        probs = self.inspect_fn(data_x, mask, cmask, xlen)

                probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                        cmask.reshape((cmask.shape[0], cmask.shape[1], 1))

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
        else:
            if self.model.use_reinforce:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_pre, read_weights_pre, \
                        write_weights_samples, read_weights_samples, probs = \
                            self.inspect_fn(data_x, mask, cmask, xlen)

                probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                        cmask.reshape((cmask.shape[0], cmask.shape[1], 1))

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_samples": read_weights_samples,
                        "write_weights_samples": write_weights_samples,
                        "read_weights_pre": read_weights_pre,
                        "write_weights_pre": write_weights_pre,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
            else:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_pre, read_weights_pre, \
                        probs = self.inspect_fn(data_x, mask, cmask, xlen)

                probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                        cmask.reshape((cmask.shape[0], cmask.shape[1], 1))

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_pre": read_weights_pre,
                        "write_weights_pre": write_weights_pre,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}

        if qmask is not None:
            vals["qmask"] = qmask

        if cmask is not None:
            vals['cmask'] = cmask

        inspect_file = self.prefix + "_inspections.pkl"
        ensure_dir_exists(SAVE_DUMP_FOLDER)
        inspect_file = os.path.join(SAVE_DUMP_FOLDER, inspect_file)
        pkl.dump(vals, open(inspect_file, "wb"), 2)

    def evaluate(self, data_gen=None, mode="Valid"):
        logger.info("Evaluating the model for %s." % mode)
        costs = []
        errors = []

        mode = mode.lower()

        if mode == "train":
            data_gen = self.train_mon_data_gen
        elif mode == "valid":
            data_gen = self.valid_data_gen
        elif mode == "test":
            data_gen = self.test_data_gen

        if self.valid_iters:
            for i in xrange(self.valid_iters):
                if self.memory_net_training:
                    try:
                        batch = next(data_gen)
                        vdata_x, vdata_q, vdata_y, cmask, mask = batch['x'], batch['q'], \
                                batch['y'], batch['cmask'], batch['mask']
                    except:
                        data_gen.reset()
                        break
                    xlen = vdata_y.shape[0]
                    cost, error = self.valid_fn(vdata_x, vdata_q, vdata_y,
                                                mask, cmask, xlen)
                else:
                    if self.use_qmask and self.model.correlation_ws:
                        try:
                            batch = next(data_gen)
                            vdata_x, vdata_y, cmask, mask, qmask = batch['x'], \
                                    batch['y'], batch['cmask'], batch['mask'], \
                                    batch['qmask']
                        except:
                            break
                        inps = OrderedDict({"X": vdata_x, "y": vdata_y,
                                            "cost_mask": cmask, "mask": mask,
                                            "qmask": qmask})
                    else:
                        try:
                            batch = next(data_gen)
                            vdata_x, vdata_y, cmask, mask = batch['x'], batch['y'], \
                                    batch['cmask'], batch['mask']
                        except:
                            break

                        inps = OrderedDict({"X": vdata_x, "y": vdata_y,
                                            "cost_mask": cmask,
                                            "mask": mask})

                    if self.pred_bow_out:
                        vbow = batch['bow_out']
                        inps.update({"bow_out": vbow})

                    xlen = vdata_y.shape[0]
                    inps.update({"seq_len": xlen})
                    cost, error = self.valid_fn(**inps)

                costs.append(cost)
                errors.append(error)
        else:
            taskerrs = defaultdict(list)
            for batch in self.valid_data_gen:
                if self.memory_net_training:
                    vdata_x, vdata_q, vdata_y, cmask, mask, task_ids = batch['x'], batch['q'], \
                            batch['y'], batch['cmask'], batch['mask'], batch['task_ids']

                    xlen = vdata_y.shape[0]
                    cost, error = self.valid_fn(vdata_x,
                                                vdata_q,
                                                vdata_y,
                                                mask,
                                                cmask,
                                                xlen)
                    if np.isnan(cost):
                        import ipdb; ipdb.set_trace()
                    taskerrs[task_ids[0]].append(error)
                else:
                    inps = OrderedDict({})
                    if self.use_qmask and self.model.correlation_ws:
                        vdata_x, vdata_y, cmask, mask, qmask, task_ids = batch['x'], \
                                batch['y'], batch['cmask'], batch['mask'], \
                                batch['qmask'], batch['task_ids']
                        inps = OrderedDict({"X": vdata_x, "y": vdata_y, "mask": mask,
                                            "cost_mask": cmask,
                                            "qmask": qmask})
                    else:
                        vdata_x, vdata_y, cmask, mask, task_ids = batch['x'], batch['y'], \
                                batch['cmask'], batch['mask'], batch['task_ids']
                        inps = OrderedDict({"X": vdata_x, "y": vdata_y, "mask": mask,
                                            "cost_mask": cmask})

                    xlen = vdata_y.shape[0]
                    inps.update({"seq_len": xlen})
                    cost, error = self.valid_fn(**inps)

                    if task_ids.ndim == 2:
                        taskerrs[task_ids[0][0]].append(error)
                    else:
                        taskerrs[task_ids[0]].append(error)

                costs.append(cost)
                errors.append(error)

            if self.report_task_errors:
                for k, v in taskerrs.iteritems():
                    print "Task %d, error is: %f." % (k, np.mean(v))

        error = np.mean(errors)
        cost = np.mean(costs)

        valid_str_errors = "%s errors: %f costs: %f"
        valid_str_errors_vals = (mode, error, cost)

        self.stats['%s_errors' % mode].append(error)
        logger.info(valid_str_errors % valid_str_errors_vals)

        self.stats['%s_cost' % mode].append(cost)
        if self.state is not None:
            self.state['%s_cost' % mode] = cost
            self.state['%s_error' % mode] = error
            self.state.val_cost = cost
            self.state.val_error = error

        if mode == "test":
            if self.is_best_valid:
                self.best_test_cost = abs(cost)
                self.best_test_error = error
                self.stats['best_cost'] = self.best_cost
                self.stats['best_error'] = self.best_error
                self.stats['best_test_cost'] = self.best_test_cost
                self.stats['best_test_error'] = self.best_test_error
                self.stats['train_cnt'] = self.cnt

            logger.info("\t>>>Best test error %f, best test cost: %f" % (self.best_test_error,
                                                                    self.best_test_cost))
        elif mode == "valid":
            self.is_best_valid = False
            if abs(cost) <= self.best_cost or error <= self.best_error:
                logger.info("Saving the best model.")
                self.best_cost = abs(cost)
                self.best_error = abs(error)
                self.save(mdl_name=self.best_mdl_name,
                        stats_name=self.best_stats_name)
                self.is_best_valid = True
            logger.info(">>>Best valid error %f, best valid cost %f" % (self.best_error,
                                                                        self.best_cost))

            if self.linear_start:
                if error - self.prev_error > self.inc_ratio:
                    if self.cur_patience > self.patience:
                        self.model.out_layer.init_params()
                        self.model.add_noise_to_params()
                        self.cur_patience -= 1
                else:
                    self.cur_patience += 1

                self.prev_error = error

            logger.info("The norm of the parameters are : ")
            self.model.params.print_param_norms()

            if self.monitor_grad_norms:
                self.print_train_grad_norms()


class NTMToyMainLoop(MainLoop):
    """
        MainLoop for the NTM toy tasks.
    """
    def __init__(self, model,
                 learning_rate=1e-3,
                 print_every=None,
                 checkpoint_every=None,
                 inspect_every=None,
                 validate_every=None,
                 train_data_gen=None,
                 valid_data_gen=None,
                 test_data_gen=None,
                 train_mon_data_gen=None,
                 reload_model=False,
                 inspect_only=False,
                 valid_only=False,
                 inspect_model=None,
                 valid_iters=None,
                 monitor_grad_norms=True,
                 monitor_full_train=False,
                 dice_inc=0.0015,
                 use_qmask=None,
                 max_iters=None,
                 state=None,
                 prefix=None):

        assert prefix is not None, "Prefix should not be empty."
        logger.info("Building the computational graph.")

        super(NTMToyMainLoop, self).__init__(model=model,
                                            learning_rate=learning_rate,
                                            checkpoint_every=checkpoint_every,
                                            print_every=print_every,
                                            inspect_every=inspect_every,
                                            inspect_only=inspect_only,
                                            valid_only=valid_only,
                                            monitor_full_train=monitor_full_train,
                                            train_data_gen=train_data_gen,
                                            train_mon_data_gen=train_mon_data_gen,
                                            test_data_gen=test_data_gen,
                                            max_iters=max_iters,
                                            valid_data_gen=valid_data_gen,
                                            validate_every=validate_every,
                                            reload_model=reload_model,
                                            prefix=prefix)

        self.dice_inc = dice_inc

        self.monitor_grad_norms = monitor_grad_norms

        self.mdl_name = self.prefix + "_model_params.pkl"
        self.stats_name = self.prefix + "_stats.pkl"
        self.best_mdl_name = self.prefix + "_best_model_params.pkl"
        self.best_stats_name = self.prefix + "_best_stats.pkl"
        self.valid_iters = valid_iters
        self.use_qmask = use_qmask

        self.stats = defaultdict(list)
        self.use_reinforce = self.model.use_reinforce
        self.use_reinforce_baseline = self.model.use_reinforce_baseline
        self.state = state
        self.max_iters = max_iters
        self.train_fn = None
        self.valid_fn = None
        self.cnt = 0
        self.is_best_valid = False
        self.prepare_model()
        self.trainpartitioner = self.model.trainpartitioner
        self.comp_grad_fn = self.model.comp_grad_fn

    def print_train_grad_norms(self):
        logger.info("Printing out the norms")

        for gm in self.trainpartitioner.gs:
            name = gm.name
            norm = ((gm.get_value()**2).sum())**.5
            logger.info("%s : %.3f" % (name, norm))

    def train(self):

        batch = self.train_data_gen.next()

        tdata_x = batch['x']
        tdata_y = batch['y']
        tcmask = batch['cmask']
        tmask = batch['mask']
        xlen = tdata_y.shape[0]
        avg_gnorm = 0.
        avg_cost = 0.
        avg_norm_up = 0.
        dice_val = 1. / (1 + self.cnt * self.dice_inc)**0.5

        inps = OrderedDict({'X': tdata_x, 'y': tdata_y, 'mask': tmask,
                            'cost_mask': tcmask})
        if self.use_qmask:
            qmask = batch['qmask']
            inps['qmask'] = qmask

        if self.comp_grad_fn:
            self.trainpartitioner.accum_grads(self.comp_grad_fn, inps)

        inps.update({'seq_len': xlen})

        if self.use_reinforce:
            outs = self.train_fn(**inps)
            cost, gnorm, norm_up, param_norm = outs[0], outs[1], \
                    outs[2], outs[3]
            idx = 3
            read_const, baseline, read_policy, write_policy = outs[idx], \
                    outs[idx+1], outs[idx+2], outs[idx+3]

            if not self.use_reinforce_baseline:
                 center, cost_std, base_reg = outs[idx+4], outs[idx+5], \
                         outs[idx+6]
        else:
            cost, gnorm, norm_up, param_norm = self.train_fn(**inps)

        if self.model.use_dice_val:
            self.model.dice_val.set_value(np.float32(dice_val))

        avg_cost = 0.9 * avg_cost + 0.1 * cost
        avg_gnorm = 0.9 * avg_gnorm +  0.1 * gnorm
        avg_norm_up = 0.9 * avg_norm_up + 0.1 * norm_up

        if self.cnt % self.print_every == 0:
            self.stats['train_cnt'] = self.cnt
            self.stats['train_cost'].append(cost)
            self.stats['norm_up'].append(norm_up)
            self.stats['param_norm'].append(param_norm)
            self.stats['gnorm'].append(gnorm)

            if self.use_reinforce:
                self.stats['read_const'].append(read_const)
                self.stats['baseline'].append(baseline)
                self.stats['read_policy'].append(read_policy)
                self.stats['write_policy'].append(write_policy)

                if self.model.use_reinforce_baseline:
                    train_str = ("Iter %d: cost: %f, update norm: %.4f, "
                                 " parameter norm: %.4f, "
                                 " norm of gradients: %.3f"
                                 " read constr %f"
                                 " baseline %f"
                                 " read policy %.3f"
                                 " write policy %.3f"
                                 " dice val %.3f")

                    train_str_vals = (self.cnt, cost.mean(), norm_up,
                                      param_norm, gnorm, read_const,
                                      baseline, read_policy,
                                      write_policy,
                                      dice_val)
                else:
                    self.stats["center"].append(center)
                    self.stats["cost_std"].append(cost_std)
                    self.stats["base_reg"].append(base_reg)

                    train_str = ("Iter %d: cost: %.4f,"
                                 " update norm: %.3f,"
                                 " parameter norm: %.4f,"
                                 " norm of grads: %.3f"
                                 " read constr %f"
                                 " baseline %f"
                                 " center %.3f"
                                 " cost_std %.3f"
                                 " base_reg %.3f"
                                 " read poli %.3f"
                                 " write poli %.3f"
                                 " dice val %.3f")

                    baseline = baseline.mean()

                    train_str_vals = (self.cnt, cost,
                                      norm_up, param_norm,
                                      gnorm, read_const, baseline.mean(),
                                      center, cost_std, base_reg,
                                      read_policy.mean(),
                                      write_policy, dice_val)

            else:
                train_str = ("Iter %d: cost: %f, update norm: %.4f, "
                             " parameter norm: %.4f, "
                             " norm of gradients: %.3f")

                train_str_vals = (self.cnt, cost, norm_up,
                                  param_norm, gnorm)
            logger.info(train_str % train_str_vals)

    def inspect_model(self, data_x=None,
                      mask=None, cmask=None,
                      qmask=None):

        if data_x is None or mask is None:
            self.valid_data_gen.reset()
            batch = self.valid_data_gen.next()
            data_x = batch['x']
            mask = batch['mask']
            cmask = batch['cmask']
            if self.use_qmask:
                qmask = batch['qmask']

        xlen = mask.shape[0]

        logger.info("Inspecting the model.")
        if not self.model.smoothed_diff_weights:
            if self.model.use_reinforce:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_samples, read_weights_samples, probs = \
                            self.inspect_fn(data_x, mask, cmask, xlen)

                probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                        cmask.reshape((cmask.shape[0], cmask.shape[1], 1))

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_samples": read_weights_samples,
                        "write_weights_samples": write_weights_samples,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
            else:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        probs = self.inspect_fn(data_x, mask, cmask, xlen)

                probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                        cmask.reshape((cmask.shape[0], cmask.shape[1], 1))

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
        else:
            if self.model.use_reinforce:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_pre, read_weights_pre, \
                        write_weights_samples, read_weights_samples, probs = \
                            self.inspect_fn(data_x, mask, cmask, xlen)

                probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                        cmask.reshape((cmask.shape[0], cmask.shape[1], 1))

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_samples": read_weights_samples,
                        "write_weights_samples": write_weights_samples,
                        "read_weights_pre": read_weights_pre,
                        "write_weights_pre": write_weights_pre,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
            else:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_pre, read_weights_pre, \
                        probs = self.inspect_fn(data_x, mask, cmask, xlen)

                probs = probs.reshape((cmask.shape[0], cmask.shape[1], -1)) * \
                        cmask.reshape((cmask.shape[0], cmask.shape[1], 1))

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_pre": read_weights_pre,
                        "write_weights_pre": write_weights_pre,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}

        if qmask is not None:
            vals["qmask"] = qmask

        if cmask is not None:
            vals['cmask'] = cmask

        inspect_file = self.prefix + "_inspections.pkl"
        ensure_dir_exists(SAVE_DUMP_FOLDER)
        inspect_file = os.path.join(SAVE_DUMP_FOLDER, inspect_file)
        pkl.dump(vals, open(inspect_file, "wb"), 2)

    def evaluate(self, data_gen=None, mode="Valid"):
        logger.info("Evaluating the model for %s." % mode)
        costs = []

        mode = mode.lower()

        if mode == "train":
            data_gen = self.train_mon_data_gen
        elif mode == "valid":
            data_gen = self.valid_data_gen
        elif mode == "test":
            data_gen = self.test_data_gen

        if self.valid_iters:
            for i in xrange(self.valid_iters):
                try:
                     batch = next(data_gen)
                     vdata_x, vdata_y, cmask, mask = batch['x'], batch['y'], \
                             batch['cmask'], batch['mask']
                except:
                    break

                inps = OrderedDict({"X": vdata_x, "y": vdata_y,
                                    "cost_mask": cmask,
                                    "mask": mask})

                if self.use_qmask:
                    qmask = batch['qmask']
                    inps['qmask'] = qmask

                xlen = vdata_y.shape[0]
                inps.update({"seq_len": xlen})
                cost = self.valid_fn(**inps)
                costs.append(cost)
        else:
            taskerrs = defaultdict(list)
            for batch in self.valid_data_gen:
                inps = OrderedDict({})
                vdata_x, vdata_y, cmask, mask = batch['x'], batch['y'], \
                            batch['cmask'], batch['mask']

                inps = OrderedDict({"X": vdata_x, "y": vdata_y, "mask": mask,
                                        "cost_mask": cmask})
                if self.use_qmask:
                    qmask = batch['qmask']
                    inps['qmask'] = qmask

                xlen = vdata_y.shape[0]
                inps.update({"seq_len": xlen})
                cost = self.valid_fn(**inps)
                costs.append(cost)

        cost = np.mean(costs)
        cost_std = np.std(costs)

        valid_str_errors = "%s costs: %f"
        valid_str_errors_vals = (mode, cost)

        logger.info(valid_str_errors % valid_str_errors_vals)
        self.stats['%s_cost_std' % mode].append(cost_std)
        self.stats['%s_cost' % mode].append(cost)
        if self.state is not None:
            self.state['%s_cost' % mode] = cost

        if mode == "test":
            if self.is_best_valid:
                self.best_test_cost = abs(cost)
                self.stats['best_cost'] = self.best_cost
                self.stats['best_test_cost'] = self.best_test_cost
                self.stats['train_cnt'] = self.cnt
            logger.info("\t>>>Best test cost: %f" % (self.best_test_cost))
        elif mode == "valid":
            self.is_best_valid = False
            if abs(cost) <= self.best_cost:
                logger.info("Saving the best model.")
                self.best_cost = abs(cost)
                self.save(mdl_name=self.best_mdl_name,
                        stats_name=self.best_stats_name)
                self.is_best_valid = True
            logger.info(">>>Best valid cost %f" % (self.best_cost))

            logger.info("The norm of the parameters are : ")
            self.model.params.print_param_norms()

            if self.monitor_grad_norms:
                self.print_train_grad_norms()

class SeqMNISTMainLoop(MainLoop):
    """
        MainLoop for the NTM toy tasks.
    """
    def __init__(self,
                 model,
                 learning_rate=1e-3,
                 print_every=None,
                 checkpoint_every=None,
                 inspect_every=None,
                 validate_every=None,
                 train_data_gen=None,
                 valid_data_gen=None,
                 test_data_gen=None,
                 num_epochs=100,
                 train_mon_data_gen=None,
                 reload_model=False,
                 inspect_only=False,
                 valid_only=False,
                 inspect_model=None,
                 monitor_grad_norms=True,
                 monitor_full_train=False,
                 dice_inc=0.0015,
                 use_qmask=None,
                 state=None,
                 prefix=None):

        assert prefix is not None, "Prefix should not be empty."
        logger.info("Building the computational graph.")

        super(SeqMNISTMainLoop, self).__init__(model=model,
                                               learning_rate=learning_rate,
                                               checkpoint_every=checkpoint_every,
                                               print_every=print_every,
                                               inspect_every=inspect_every,
                                               inspect_only=inspect_only,
                                               valid_only=valid_only,
                                               monitor_full_train=monitor_full_train,
                                               train_data_gen=train_data_gen,
                                               train_mon_data_gen=train_mon_data_gen,
                                               test_data_gen=test_data_gen,
                                               valid_data_gen=valid_data_gen,
                                               validate_every=validate_every,
                                               reload_model=reload_model,
                                               prefix=prefix)

        self.dice_inc = dice_inc
        self.num_epochs = num_epochs
        self.monitor_grad_norms = monitor_grad_norms
        self.mdl_name = self.prefix + "_model_params.pkl"
        self.stats_name = self.prefix + "_stats.pkl"
        self.best_mdl_name = self.prefix + "_best_model_params.pkl"
        self.best_stats_name = self.prefix + "_best_stats.pkl"
        self.avg_gnorm = 0.
        self.avg_cost = 2.3

        self.avg_norm_up = 0.
        self.avg_pnorm = 0.
        self.avg_errors = 0.9

        self.stats = defaultdict(list)
        self.use_reinforce = self.model.use_reinforce
        self.use_reinforce_baseline = self.model.use_reinforce_baseline
        self.state = state
        self.train_fn = None
        self.valid_fn = None
        self.cnt = 0
        self.is_best_valid = False
        self.prepare_model()
        self.trainpartitioner = self.model.trainpartitioner
        self.comp_grad_fn = self.model.comp_grad_fn

    def print_train_grad_norms(self):
        logger.info("Printing out the norms")
        for gm in self.trainpartitioner.gs:
            name = gm.name
            norm = ((gm.get_value()**2).sum())**.5
            logger.info("%s : %.3f" % (name, norm))

    def run(self):
        logger.info("Started running the mainloop...")
        for nepoch in xrange(self.num_epochs):
            train_iter = self.train_data_gen.get_epoch_iterator()

            print "Starting training on epoch, ", nepoch
            for i, batch in enumerate(train_iter):
                self.train(batch)
                self.cnt += 1

            self.evaluate(mode="Valid")

            if self.test_data_gen:
                self.evaluate(mode="Test")

            if self.monitor_full_train:
                self.evaluate(data_gen=self.train_mon_data_gen, mode="Train")

            if (self.inspect_every is not None and
                self.cnt % self.inspect_every == 0 and self.model_inspection):
                self.inspect_model()

            if self.cnt % self.checkpoint_every == 0:
                self.save()

    def train(self, batch):
        tdata_x = batch[0].reshape((batch[0].shape[0], -1))
        tdata_y = batch[1].flatten()

        avg_gnorm = self.avg_gnorm
        avg_cost = self.avg_cost

        avg_norm_up = self.avg_norm_up
        avg_pnorm = self.avg_pnorm
        avg_errors = self.avg_errors

        dice_val = 1. / (1 + self.cnt * self.dice_inc)**0.5

        inps = OrderedDict({'X': tdata_x, 'y': tdata_y})

        xlen = tdata_x.shape[1]
        inps.update({"seq_len": xlen})


        if self.use_reinforce:
            outs = self.train_fn(**inps)
            cost, gnorm, norm_up, param_norm, errors, = outs[0], outs[1], \
                    outs[2], outs[3], outs[4]
            idx = 5
            read_const, baseline, read_policy, write_policy = outs[idx], \
                    outs[idx+1], outs[idx+2], outs[idx+3]

            if not self.use_reinforce_baseline:
                 center, cost_std, base_reg = outs[idx+4], outs[idx+5], \
                         outs[idx+6]
        else:
            cost, gnorm, norm_up, param_norm, errors = self.train_fn(**inps)

        if self.model.use_dice_val:
            self.model.dice_val.set_value(np.float32(dice_val))

        if isinstance(cost, np.ndarray):
            self.avg_cost = 0.9 * avg_cost + 0.1 * cost.mean()
        else:
            self.avg_cost = 0.9 * avg_cost + 0.1 * cost

        self.avg_gnorm = 0.9 * avg_gnorm +  0.1 * gnorm
        self.avg_norm_up = 0.9 * avg_norm_up + 0.1 * norm_up
        self.avg_errors = 0.9 * avg_errors + 0.1 * errors
        self.avg_pnorm = 0.9 * avg_pnorm + 0.1 * param_norm

        if self.cnt % self.print_every == 0:
            self.stats['train_cnt'] = self.cnt
            self.stats['train_cost'].append(cost)
            self.stats['norm_up'].append(norm_up)
            self.stats['param_norm'].append(param_norm)
            self.stats['gnorm'].append(gnorm)
            self.stats['errors'].append(errors)

            if self.use_reinforce:
                self.stats['read_const'].append(read_const)
                self.stats['baseline'].append(baseline)
                self.stats['read_policy'].append(read_policy)
                self.stats['write_policy'].append(write_policy)

                if self.model.use_reinforce_baseline:
                    train_str = ("Iter %d: cost: %f, update norm: %.4f, "
                                 " parameter norm: %.4f, "
                                 " norm of gradients: %.3f"
                                 " read constr %f"
                                 " baseline %f"
                                 " read policy %.3f"
                                 " write policy %.3f"
                                 " dice val %.3f"
                                 " errors %.4f")

                    train_str_vals = (self.cnt, avg_cost.mean() if isinstance(avg_cost,
                                                                              np.ndarray) else avg_cost, avg_norm_up,
                                      avg_pnorm, avg_gnorm, read_const,
                                      baseline, read_policy, write_policy,
                                      dice_val, avg_errors)
                else:
                    self.stats["center"].append(center)
                    self.stats["cost_std"].append(cost_std)
                    self.stats["base_reg"].append(base_reg)

                    train_str = ("Iter %d: cost: %.4f,"
                                 " update norm: %.3f,"
                                 " parameter norm: %.4f,"
                                 " norm of grads: %.3f"
                                 " read constr %f"
                                 " baseline %f"
                                 " center %.3f"
                                 " cost_std %.3f"
                                 " base_reg %.3f"
                                 " read poli %.3f"
                                 " write poli %.3f"
                                 " dice val %.3f",
                                 " errors %.3f")

                    baseline = baseline.mean()
                    train_str_vals = (self.cnt, avg_cost, avg_norm_up, avg_pnorm,
                                      avg_gnorm, read_const, baseline,
                                      center, cost_std, base_reg, read_policy,
                                      write_policy, dice_val, avg_errors)

            else:
                train_str = ("Iter %d: cost: %f, update norm: %.4f, "
                             " parameter norm: %.4f, "
                             " norm of gradients: %.3f",
                             " errors: %.3f")

                train_str_vals = (self.cnt, avg_cost, avg_norm_up,
                                  avg_pnorm, avg_gnorm, avg_errors)

            logger.info("".join(list(train_str)) % train_str_vals)

    def inspect_model(self, data_x=None,
                      mask=None, cmask=None,
                      qmask=None):

        if data_x is None or mask is None:
            self.valid_data_gen.reset()
            batch = self.valid_data_gen.get_epoch_iterator().next()
            data_x = batch[0].reshape((batch[0].shape[0], -1))

        logger.info("Inspecting the model.")
        if not self.model.smoothed_diff_weights:
            if self.model.use_reinforce:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_samples, read_weights_samples, probs = \
                            self.inspect_fn(data_x, mask, cmask)

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_samples": read_weights_samples,
                        "write_weights_samples": write_weights_samples,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
            else:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        probs = self.inspect_fn(data_x, mask, cmask)

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
        else:
            if self.model.use_reinforce:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_pre, read_weights_pre, \
                        write_weights_samples, read_weights_samples, probs = \
                            self.inspect_fn(data_x, mask, cmask)

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_samples": read_weights_samples,
                        "write_weights_samples": write_weights_samples,
                        "read_weights_pre": read_weights_pre,
                        "write_weights_pre": write_weights_pre,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}
            else:
                h_t, m_t, mem_read_t, write_weights, read_weights, \
                        write_weights_pre, read_weights_pre, \
                        probs = self.inspect_fn(data_x)

                vals = {"h_t": h_t,
                        "m_t": m_t,
                        "qmask": qmask,
                        "mem_read": mem_read_t,
                        "read_weights": read_weights,
                        "write_weights": write_weights,
                        "read_weights_pre": read_weights_pre,
                        "write_weights_pre": write_weights_pre,
                        "probs": probs,
                        "data_x": data_x,
                        "mask": mask}

        inspect_file = self.prefix + "_inspections.pkl"
        ensure_dir_exists(SAVE_DUMP_FOLDER)
        inspect_file = os.path.join(SAVE_DUMP_FOLDER, inspect_file)
        pkl.dump(vals, open(inspect_file, "wb"), 2)

    def evaluate(self, data_gen=None, mode="Valid"):
        logger.info("Evaluating the model for %s." % mode)
        costs = []
        errors = []
        mode = mode.lower()

        if mode == "train":
            data_gen = self.train_mon_data_gen.get_epoch_iterator()
        elif mode == "valid":
            data_gen = self.valid_data_gen.get_epoch_iterator()
        elif mode == "test":
            data_gen = self.test_data_gen.get_epoch_iterator()

        for batch in data_gen:
            vdata_x, vdata_y = batch[0], batch[1]
            vdata_x = vdata_x.reshape((vdata_x.shape[0], -1))
            vdata_y = vdata_y.flatten()
            inps = OrderedDict({"X": vdata_x, "y": vdata_y})

            xlen = vdata_x.shape[1]
            inps.update({"seq_len": xlen})

            cost, error = self.valid_fn(**inps)
            costs.append(cost)
            errors.append(error)

        cost = np.mean(costs)
        error = np.mean(errors)

        valid_str_errors = "%s errors: %f"
        valid_str_errors_vals = (mode, error)

        valid_str_cost = "%s costs: %f"
        valid_str_cost_vals = (mode, cost)

        logger.info(valid_str_errors % valid_str_errors_vals)
        logger.info(valid_str_cost % valid_str_cost_vals)

        self.stats['%s_cost' % mode].append(cost)
        self.stats['%s_error' % mode].append(error)

        if self.state is not None:
            self.state['%s_cost' % mode] = cost
            self.state['%s_cost' % mode] = cost

        if mode == "test":
            if self.is_best_valid:
                self.best_test_cost = abs(cost)
                self.best_test_error = error
                self.stats['best_cost'] = self.best_cost
                self.stats['best_error'] = self.best_error
                self.stats['best_test_cost'] = self.best_test_cost
                self.stats['best_test_error'] = self.best_test_error
                self.stats['train_cnt'] = self.cnt
                logger.info("\t>>>Best test cost: %f best test error %f" % (self.best_test_cost, self.best_test_error))
        elif mode == "valid":
            self.is_best_valid = False
            if abs(cost) <= self.best_cost or error <= self.best_error:
                logger.info("Saving the best model.")
                self.best_cost = abs(cost)
                self.best_error = abs(error)
                self.save(mdl_name=self.best_mdl_name,
                        stats_name=self.best_stats_name)
                self.is_best_valid = True

            logger.info(">>>Best valid cost %f best valid error %f" % (self.best_cost,
                                                                       self.best_error))
            logger.info("The norm of the parameters are : ")
            self.model.params.print_param_norms()
