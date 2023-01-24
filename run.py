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
from torch_utils import get_opt, get_scheduler, load, get_max_mem_c
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
	model_fn, p, b = make_functional_with_buffers(model)
	model_fn_vmap = vmap(model_fn, in_dims=(None, None, 0))

	opt = get_opt(**c.opt.d_flat)(model.parameters())
	scheduler = get_scheduler(n_step=c.n_step, **c.scheduler.d_flat)(opt)

	if c.lo_ve_path:
		c, model, opt, r = load(c, things_to_load=dict(model=model, opt=opt, r=r))

	model, opt, scheduler = c.distribute.prepare(model, opt, scheduler)

	if 'train' in c.mode:
		model.train()
	elif c.mode=='eval':
		model.eval()

	return model, model_fn_vmap, opt, scheduler

def loss_fn(
	step: int,
	r: torch.Tensor=None,
	deltar: torch.Tensor=None,
	model:torch.nn.Module=None,
	_model_fn: Callable=None,
	mode: str = 'train', 
	**kw, 
):

	with torch.no_grad():
		r, acc, deltar = sample_b(model, r, deltar, n_corr=c.data.n_corr)
		
		if step < c.n_pre_step:
			center_points = get_center_points(c.data.n_e, c.data.a)
			r = keep_around_points(r, center_points, l=5.+10.*step/c.n_pre_step)
		
		pe = compute_pe_b(r, c.data.a, c.data.a_z)	
	
	ke = compute_ke_b(model, _model_fn, r, ke_method=c.model.ke_method).detach().requires_grad_(False)
	
	with torch.no_grad():
		e = pe + ke
		e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
		e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)
		energy = (e_clip - e_clip.mean())
		
	loss, grads, params = None, None, None

	if 'train' in mode:
		loss = (energy * model(r)).mean()
		# c.distribute.backward(loss)
	
	v_init = dict(r=r, deltar=deltar)

	return loss, v_init, dict(r=r, deltar=deltar, e=e, pe=pe, ke=ke, acc=acc, loss=loss)

from copy import deepcopy

def detach_tree(v_d: dict):
	items = []
	for k, v in v_d.items():

		if isinstance(v, list):
			v = {k + '_' + str(i): sub_i for i, sub_i in enumerate(v)}
		
		if isinstance(v, dict):
			v = {k: detach_tree(v)}
		elif isinstance(v, torch.Tensor):
			v = v.detach()
		
		items += [(k, v),]
	
	return dict(items)
	
from pyfig_utils import Sub

def run(c: Pyfig=None, c_update: dict=None, **kw):

	model, model_fn_vmap, opt, scheduler = init_exp(c, c_update, **kw)

	center_points 	= get_center_points(c.data.n_e, c.data.a)
	r 				= center_points + torch.randn(size=(c.data.n_b, *center_points.shape), dtype=c.dtype, device=c.distribute.device)
	r.requires_grad_(False)
	deltar			= torch.tensor([0.02,], device=c.distribute.device, dtype=c.dtype)
	
	v_init = dict(r=r, deltar=deltar)
	compute_loss = partial(loss_fn, mode=c.mode, _model_fn=model_fn_vmap, model=model)
	debug_dict(msg='pyfig:run:preloop = \n', c_init=c.d)

	class Metrix(Sub):
		_t0: 		   float = time.time()
		max_mem_alloc: float = None
		t_per_it: 	   float = None
		step: 		   int 	 = 0

		exp_stats: 		list = ['max_mem_alloc', 't_per_it']
		source: 		str  = 'exp_stats/'

		def __init__(ii, parent=None):
			super().__init__(parent)
			torch.cuda.reset_peak_memory_stats()

		def tick(ii, step: int, **kw) -> dict:
			dstep = step - ii.step 

			_t1 = time.time()
			ii.t_per_it = (_t1 - ii._t0) / float(dstep)
			ii._t0 = time.time()

			ii.max_mem_alloc = torch.cuda.max_memory_allocated() // 1024 // 1024
			torch.cuda.reset_peak_memory_stats()
			return dict(exp_stats = {k: getattr(ii, k) for k in ii.exp_stats})
		
		def to_dict(ii):
			return {k: getattr(ii, k) for k in ii.exp_stats}

	metrix = Metrix()

	c.start()
	for rel_step, step in enumerate(range(1, (c.n_step if 'train' in c.mode else c.n_eval_step) + 1)):

		model.zero_grad(set_to_none=True)

		loss, v_init, v_d = compute_loss(step, **v_init)


		if 'train' in c.mode:
			c.distribute.backward(loss)
			v_d['grads'] = {k:p.grad.detach() for k,p in model.named_parameters()}
		
		if (not (step % c.distribute.sync_step)) and (c.resource.n_gpu > 1):
			v_d = c.distribute.sync(step, v_d)


		if 'train' in c.mode:
			with torch.no_grad():
				for k, p in model.named_parameters():
					p.grad.copy_(v_d.get('grads').get(k))
				opt.step()
				scheduler.step()


		###--- cpu only from here ---###
		if int(c.distribute.rank)==0 and c.distribute.head:

			if ('eval' in c.mode) or not (step % c.log_metric_step):
				v_d['grads'] = {k:p.grad.detach().cpu().numpy() for k,p in model.named_parameters()}
				v_d['params'] = {k:p.detach().cpu().numpy() for k,p in model.named_parameters()}

				v_cpu_d = npify_tree(v_d)

				t_metrix = metrix.tick(step)
				v_metrix = compute_metrix(v_cpu_d, source=c.mode, sep='/')
				wandb.log(v_metrix | t_metrix, step=step)

				if (step//c.log_metric_step)==1:
					import pprint
					c.make_a_note(c.exp_dir/'s1_metrix', pprint.pformat(v_metrix | t_metrix))


		if not (step % c.log_state_step) and 'savestate' in c.mode:
			v_d['params'] = {k:p.detach() for k,p in model.named_parameters()}
			name = f'{c.mode}_i{step}.state'
			lo_ve(path=c.state_dir/name, data=v_d)


	c.wb.run.finish()
	torch.cuda.empty_cache()

	if 'eval' in c.mode:
		api = wandb.Api()
		print(str(c.wb.wb_run_path))
		run = api.run(str(c.run_id))
		post_process(run)
		run.finish()

	return npify_tree(v_cpu_d) | metrix.to_dict()


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
   
			objective = partial(objective, c=c, run=run)
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
