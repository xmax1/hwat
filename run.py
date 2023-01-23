import time
import wandb
from datetime import datetime
from pathlib import Path

import traceback
from typing import Callable

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from functorch import make_functional_with_buffers, vmap
from torch_utils import get_opt, get_scheduler, load, update_model, get_max_mem_c
from utils import debug_dict, npify_tree, compute_metrix, flat_any

from pyfig_utils import Param, lo_ve
from pyfig import Pyfig 
from functools import partial

from hwat import Ansatz_fb as Model
from hwat import init_r, get_center_points
from hwat import keep_around_points, sample_b
from hwat import compute_ke_b, compute_pe_b

def post_process(run: wandb.Api):
	import numpy as np

	c: dict = run.config
	print('post_process:wb_run_c = \n', c)

	try:
		print('succes', c.a)
	except:
		pass

	history = run.scan_history(keys=['e',])
	opt_obj = np.asarray([row['e'] for row in history])
	
	a_z = np.asarray(c['a_z']).flatten()
	a = np.asarray(c['a'])

	exp_metaid = f'{c["charge"]}_{c["spin"]}_{"-".join([str(int(float(i))) for i in a_z])}_{a.mean():.2f}'
	columns = ["charge_spin_az0-az1-..._pmu", "Energy", "Error (+/- std)"]
	print('post_process:opt_obj = \n', opt_obj)
	print('post_process:columns = \n', columns)
	data = [exp_metaid, opt_obj.mean(), opt_obj.std()]
	print('post_process:data = \n', data)
	Result = wandb.Table(
		columns=columns, 
		data=data,
	)

	run.summary.update(dict(Result=Result))

	print(Result)

# def get_run(run: str|Path=None) -> wandb.run:
# 	api = wandb.Api()
# 	run = api.run(str(run))
# 	return run


def init_exp(c: Pyfig, c_update: dict=None, **kw):
	
	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()

	c_update = (c_update or {}) | (kw or {})
	c.update(c_update)
	debug_dict(msg='init_exp:c = \n', c=c.d)

	c.set_dtype()
	c.distribute.set_seed()
	c.distribute.set_device()
	c.to(device=c.distribute.device, dtype=c.dtype)

	model: torch.nn.Module = c.partial(Model).to(device=c.distribute.device, dtype=c.dtype)
	model_fn, _, buffers = make_functional_with_buffers(model)  
	model_fn_vmap = lambda params, *_v: vmap(model_fn, in_dims=(None, None, 0))(params, buffers, *_v).sum()
	
	opt = get_opt(**c.opt.d_flat)(model.parameters())
	scheduler = get_scheduler(n_step=c.n_step, **c.scheduler.d_flat)(opt)

	if c.lo_ve_path:
		c, model, opt, r = load(c, things_to_load=dict(model=model, opt=opt, r=r))

	model, opt, scheduler, buffers = c.distribute.prepare(model, opt, scheduler, buffers)

	if 'train' in c.mode:
		model.train()
	elif c.mode=='eval':
		model.eval()

	return model, model_fn_vmap, opt, scheduler

def loss_fn(
	step: int,
	r: torch.Tensor,
	deltar: torch.Tensor,
	model:torch.nn.Module,
	model_fn_vmap: Callable,
):
	# with torch.no_grad():
	
	r, acc, deltar = sample_b(model, r, deltar, n_corr=c.data.n_corr)
	if step < c.n_pre_step:
		center_points = get_center_points(c.data.n_e, c.data.a)
		r = keep_around_points(r, center_points, l=5.+10.*step/c.n_pre_step)

	for p in model.parameters():
		p.requires_grad = True
	model_rv = lambda _r: model_fn_vmap(params, _r)
	
	pe = compute_pe_b(r, c.data.a, c.data.a_z)

	ke = compute_ke_b(model, model_rv, r, ke_method=c.model.ke_method)

	e = pe.detach().requires_grad_(False) + ke.detach().requires_grad_(False)
	e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
	e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

	model.zero_grad(set_to_none=True)
	for p in model.parameters():
		p.requires_grad = True
	r_loss = r.detach()

	loss: torch.Tensor = ((e_clip - e_clip.mean())*model(r_loss)).mean()
	loss.backward()
	# c.distribute.backward(loss)

	params = {k:p for k,p in model.named_parameters()} # needs to be done here unless dis-attached from graph
	grads = {k:p.grad for k,p in params.items()}

	return (r, deltar), dict(e=e, pe=pe, ke=ke, acc=acc, grads=grads, params=params, loss=loss)

from copy import deepcopy


def detach_tree(v_d: dict):
	items = []
	for k,v in v_d.items():
		if isinstance(v, dict):
			v = detach_tree(v)
		elif isinstance(v, torch.Tensor):
			v = v.detach()
		items += [(k, v),]
	return dict(items)
	
def run(c: Pyfig=None, c_update: dict=None, **kw):

	model, model_fn_vmap, opt, scheduler = init_exp(c, c_update, **kw)

	center_points 	= get_center_points(c.data.n_e, c.data.a)
	r				= init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
	deltar			= torch.tensor([0.02,], device=c.distribute.device, dtype=c.dtype)
	
	v_d = dict(r=r, deltar=deltar)
	
	debug_dict(msg='pyfig:run:preloop = \n', c_init=c.d)

	t0 = time.time()
	c.start()
	for step in range(1, (c.n_step if 'train' in c.mode else c.n_eval_step) + 1):

		# model.requires_grad_(False)
		# model.zero_grad(set_to_none=True)
		# opt.zero_grad(set_to_none=True)

		(r, deltar), v_d = loss_fn(step, r, deltar, model, model_fn_vmap)
		# debug_dict(msg='v_d', v_d=deepcopy(v_d))

		# if (not (step % c.distribute.sync_step)) and (c.resource.n_gpu-1) and 'train' in c.mode:
		# 	v_d = c.distribute.sync(step, v_d)

		if 'train' in c.mode:
			# model.zero_grad()
			update_model(step, model, grads= v_d['grads'])
			scheduler.step()
			opt.step()

		v_d = detach_tree(v_d)
		v_cpu_d = npify_tree(v_d)

		###--- cpu only from here ---###
		if not (step % c.log_metric_step):
			print(v_cpu_d.keys())
			
			v_cpu_d['max_mem_alloc'] = torch.cuda.max_memory_allocated() // 1024 // 1024
			torch.cuda.reset_peak_memory_stats() 

			if int(c.distribute.rank)==0 and c.distribute.head:
				wandb.log(compute_metrix(v_cpu_d, mode=c.mode), step=step)

		if not (step % c.log_state_step) and 'savestate' in c.mode:
			name = f'{c.mode}_i{step}.state'
			lo_ve(path=c.state_dir/name, data=v_d)
		
		if 'eval' in c.mode:
			v_eval = dict(filter(lambda kv: kv[0] in c.eval_me, v_cpu_d.items()))
			wandb.log(compute_metrix(v_eval, mode=c.mode), step=step)

	t1 = time.time()
	v_cpu_d['t_per_it'] = (t1 - t0)/step

	c.wb.run.finish()
	torch.cuda.empty_cache()

	if 'eval' in c.mode:
		api = wandb.Api()
		run = api.run(str(c.wb.wb_run_path))
		post_process(run)
		run.finish()

	return npify_tree(v_cpu_d)

if __name__ == "__main__":
	
	allowed_mode_all = 'train:eval:max_mem:opt_hypam:profile:train-eval'

	c = Pyfig(notebook=False, sweep=None, c_update=None)

	res, v_run = dict(), dict()
	for mode in ([c.mode,] if c.mode else c.multimode.split(':')):
		c.mode = mode
		print('run.py:mode = \n ***', c.mode, '***')
		assert c.mode in allowed_mode_all

		c.update(**v_run)
		debug_dict(msg='run:c = \n', cd=c.d)

		if 'opt_hypam' in mode:
			from opt_hypam_utils import opt_hypam, objective
   
			objective = partial(objective, run=run, c=c, mode='train-eval')
			v_run = opt_hypam(objective, c)

		elif 'max_mem' in mode:
			from torch_utils import get_max_mem_c

			v_run = get_max_mem_c(run, c=c, mode='train', n_step=10*c.log_metric_step)
			
			now = datetime.now().strftime("%d-%m-%y:%H-%M-%S")
			line = now + ',' + ','.join([str(v) for v in 
				[v_run['max_mem_alloc'], c.data.n_b, c.data.n_e, c.model.n_fb, c.model.n_sv, c.model.n_pv]])
			with open('./dump/max_mem.csv', 'a+') as f:
				f.writelines([line])

		elif 'profile' in mode:
			from torch_utils import gen_profile
			v_run = gen_profile(run, profile_dir=c.profile_dir, mode='train', **v_run)

		else:
			v_run = run(c=c, **v_run)

		res[mode] = v_run
