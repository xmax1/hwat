import time
import wandb
from datetime import datetime

import traceback
from typing import Callable

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from functorch import make_functional_with_buffers, vmap
from torch_utils import get_opt, get_scheduler, load, update_model, get_max_mem_c
from utils import get_max_n_from_filename, debug_dict, numpify_tree, torchify_tree, cpuify_tree, lo_ve, compute_metrix

from pyfig_utils import Param
from pyfig import Pyfig 
from functools import partial

from hwat import Ansatz_fb as Model
from hwat import init_r, get_center_points
from hwat import keep_around_points, sample_b
from hwat import compute_ke_b, compute_pe_b


def get_run_if_wb_path(run: wandb.Api|str|Path=None) -> wandb.Run:
	if isinstance(run, str|Path):
		api = wandb.Api()
		run = api.run(str(run))
	return run


def init_exp(c: Pyfig, c_init: dict=None, **kw):
	
	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()

	c_init = (c_init or {}) | (kw or {})
	debug_dict(msg='init_exp:c_init: ', c_init=c_init)
	c.update(c_init)

	c.set_dtype()
	c.distribute.set_seed()
	c.distribute.set_device()
	c.to(device=c.distribute.device, dtype=c.dtype)

	model: torch.nn.Module = c.partial(Model).to(device=c.distribute.device, dtype=c.dtype)
	model_fn, params, buffers = make_functional_with_buffers(model)  
	model_fn_vmap = lambda params, *_v: vmap(model_fn, in_dims=(None, None, 0))(params, buffers, *_v).sum()
	
	opt = get_opt(**c.opt.d_flat)(model.parameters())
	scheduler = get_scheduler(n_step=c.n_step, **c.scheduler.d_flat)(opt)

	if c.lo_ve_path:
		c, model, opt, r = load(c, things_to_load=dict(model=model, opt=opt, r=r))

	model, opt, scheduler, buffers = c.distribute.prepare(model, opt, scheduler, buffers)

	if c.mode=='train':
		model.train()
	elif c.mode=='evaluate':
		model.eval()

	return model, model_fn_vmap, opt, scheduler

def loss_fn(
	step: int, 
	v_d: dict,
	model:torch.nn.Module,
	model_fn_vmap: Callable
):
	with torch.no_grad(): # does not affect fns
		v_d |= sample_b(model, v_d['r'], v_d['deltar'], n_corr=c.data.n_corr)
		if step < c.n_pre_step:
			center_points = get_center_points(c.data.n_e, c.data.a)
			r = keep_around_points(v_d['r'], center_points, l=5.+10.*step/c.n_pre_step)

		model_rv = lambda _r: model_fn_vmap(model.parameters(), _r)
		pe = compute_pe_b(v_d['r'], c.data.a, c.data.a_z)
		ke = compute_ke_b(model, model_rv, v_d['r'], ke_method=c.model.ke_method)
		e = pe + ke
		e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
		e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)


	if c.mode=='train':

		model.requires_grad_(True)
		loss: torch.Tensor = ((e_clip - e_clip.mean())*model(v_d['r'])).mean()
		c.distribute.backward(loss)
  
		params = {k:p for k,p in model.named_parameters()}
		grads = {k:p.grad for k,p in params.items()}

		v_d |= dict(grads=grads, params=params, loss=loss)  

	return v_d | dict(e=e, pe=pe, ke=ke, opt_obj=e.mean())

def run(c: Pyfig, c_init: dict=None, **kw):

	model, model_fn_vmap, opt, scheduler = init_exp(c, c_init, **kw)

	center_points 	= get_center_points(c.data.n_e, c.data.a)
	r				= init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
	deltar			= torch.tensor([0.02,], device=c.distribute.device, dtype=c.dtype)
	v_d = dict(r=r, deltar=deltar, loop_vars=['r', 'deltar', 'loop_vars'])
	
	t0 = time.time()
	c.start()
	for step in range(1, (c.n_step if c.mode=='train' else c.n_eval_step) + 1):

		model.requires_grad_(False)
		model.zero_grad(set_to_none=True)

		v_d: dict = loss_fn(step, v_d, model, model_fn_vmap)
  
		if (not (step % c.distribute.sync_step)) and (c.resource.n_gpu-1) and mode=='train':
			v_d = c.distribute.sync(step, v_d)

		if c.mode=='train':
			update_model(model, grads= v_d.get('grads'), step=step)
			opt.step()
			scheduler.step()

		with torch.no_grad():

			v_cpu_d = cpuify_tree(v_d)
			v_d = {k:v for k,v in v_d.items() if k in v_d['loop_vars']}

			###--- cpu only from here ---###
			if not (step % c.log_metric_step):

				v_cpu_d['max_mem_alloc'] = torch.cuda.max_memory_allocated() // 1024 // 1024
				torch.cuda.reset_peak_memory_stats() 
	
				t_diff, t0 = time.time() - t0, time.time()
				v_cpu_d['t_per_it'] = t_diff/c.distribute.sync_step

				if int(c.distribute.rank)==0 and c.distribute.head:
					metrix = compute_metrix(v_cpu_d, mode=c.mode)
					debug_dict(msg='metrix', metrix=metrix, step=step//c.log_metric_step)
					wandb.log(metrix, step=step)
					
	 
			if not (step % c.log_state_step):
				name = f'{c.mode}_i{step}.state'
				lo_ve(path=c.state_dir/name, data=v_d)
	
	
	
	torch.cuda.empty_cache()
	# run = get_run_if_wb_path(run)
	post_process(ii.wb.run)
	run.finish()


	def post_process(run: wandb.Api):
		import numpy as np

		c: dict = run.config
		history = run.scan_history(keys=['e'])
		opt_obj = [row['e'] for row in history]
  
		a_z = np.asarray(c['a_z']).squeeze()
		a = np.asarray(c['a']).squeeze()
		exp_metaid = f'{c.charge_c.spin_("-".join([int(i) for i in a_z]))_a.mean()}'
		Result = wandb.Table(
      				columns=["charge_spin_az0-az1-..._pmu", "Energy", "Error (+/- std)"], 
                    data=[exp_metaid, opt_obj.mean(), opt_obj.std()])

		run.summary.update(dict(Result=Result))
	
	return numpify_tree(v_cpu_d)


if __name__ == "__main__":

	c = Pyfig(notebook=False, sweep=None, c_init=None)
 
	res = dict()
	v_run = dict()
	for mode in ([c.mode,] if c.mode else c.multimode.split(':')):
		c.mode = mode

		if mode == 'opt_hypam':
			from opt_hypam_utils import get_hypam_from_study, opt_hypam, objective
   
			objective = partial(objective, c=c, run=run)
			v_run = opt_hypam(objective, c)
			debug_dict(v_run, msg='opt_hypam:v_run')

		elif mode=='max_mem':
			v_run = get_max_mem_c(run, mode='train', n_step=2*c.log_metric_step)
			now = datetime.now().strftime("%d-%m-%y:%H-%M-%S")
			line = now + ',' + ','.join([str(v) for v in 
							[v_run['max_mem_alloc'], c.data.n_b, c.data.n_e, c.model.n_fb, c.model.n_sv, c.model.n_pv]])
			with open('./dump/mem.csv', 'a+') as f:
				f.writelines([line])

		elif mode=='profile':
			from torch_utils import gen_profile
			v_run = gen_profile(run, profile_dir=c.profile_dir, mode='train', **v_run)
		else:
			print('train or evaluate')

			v_run = run(c=c, **v_run)

		c.update(**v_run)
		res[mode] = v_run


	
	# 	system_metrics = run.history(stream="events")
	# 	system_metrics.to_csv("sys_metrics.csv")
	# 	# if run.state == "finished":
	# 	row["_timestamp"]
	# 	# for i, row in run.history().iterrows():

	# 	for i, row in run.history(keys=["accuracy"]).iterrows():
	#   print(row["_timestamp"], row["accuracy"])
