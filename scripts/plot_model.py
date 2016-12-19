import cPickle as pkl
import pylab

insps = pkl.load(open("ntm_on_fb_BABI_task_all_addr10_no_learn_h0_l1_no_2read_steps_quad_interactions_mem_tanh_2rheads_1rsteps_l2pen_simple_small_10k_hard_nogru2q_task2_loc_addr_now2v_qmask_small_address_2_sw_avgh0_grucont_inspections.pkl"))
rws = insps['read_weights_samples'][:, : , 0, :].sum(1) * insps['mask'][:, 0, None]
wws = insps['write_weights_samples'][:, 0, :] * insps['mask'][:, 0, None]
pylab.matshow(wws.T[:, :-1], cmap="gray")
pylab.title("wws")
pylab.matshow(rws.T[:, 1:], cmap="gray")
pylab.title("rws")
pylab.show()
