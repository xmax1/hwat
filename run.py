# TORCH MNIST DISTRIBUTED EXAMPLE

"""run.py:"""
#!/usr/bin/env python
import os
from accelerate import Accelerator
import torch
from typing import Callable
import time
from accelerate.utils import set_seed
import numpy as np
import pprint
from pathlib import Path
from time import sleep
import shutil
from functools import partial
import optree
from copy import deepcopy
import wandb
from utils import check
# https://huggingface.co/docs/accelerate/basic_tutorials/migration
from pyfig import Pyfig
from torch import nn
from hwat import sample_b, compute_ke_b

from hwat import compute_ke_b, compute_pe_b
from hwat import init_r, get_center_points
from functorch import make_functional_with_buffers, vmap
from copy import deepcopy


from optuna import Trial
import optuna
from optuna import pruners, samplers
from pyfig_utils import lo_ve
import sys
from pyfig_utils import Param

from torch import nn
 
def init_agent():
	pprint.pprint(run.config)
	c = Pyfig(init_arg=run.config)
	return c

def suggest_hypam(trial: Trial, name: str, v: Param):
	if isinstance(v, dict):
		v = Param(**v)

	if not v.domain:
		return trial.suggest_categorical(name, v.values)

	if v.sample:
		if v.step_size:
			return trial.suggest_discrete_uniform(name, *v.domain, q=v.step_size)
		elif v.log:
			return trial.suggest_loguniform(name, *v.domain)
		else:
			return trial.suggest_uniform(name, *v.domain)

	if v.dtype is int:
		return trial.suggest_int(name, *v.domain, log=v.log)
	
	if v.dtype is float:
		return lambda : trial.suggest_float(name, *v.domain, log=v.log)
	
	sys.exit('{v} not supported in hypam opt')
		
def get_hypam_from_study(trial: Trial, sweep_params: dict) -> dict:
	print('trialing hypam: ')
	for i, (name, v) in enumerate(sweep_params.items()):
		v = suggest_hypam(trial, name, v)
		c_update = {name:v} if i==0 else {**c_update, name:v}
	pprint.pprint(c_update)
	return c_update

def get_max_mem_c(fn: Callable, r: torch.Tensor, deltar: torch.Tensor, start=5, **kw):
	print('finding max_mem')
	n_b_0 = r.shape[0]
	for n_b_power in range(start, 15):
		try:
			n_b = 2**n_b_power
			n_factor = max(1, n_b // n_b_0)
			r_mem = r.tile((n_factor,) + (1,)*(r.ndim-1))
			assert r_mem.shape[0] == n_b
			stats = gen_profile(fn, r=r, deltar=deltar, mode='train', c_update=dict(n_b=n_b))
			print(stats)
		except Exception as e:
			print('mem_max: ', e)
		finally:
			n_b_max = 2**(n_b_power-1)
			return dict(c_update=dict(n_b=n_b_max))

def opt_hypam(c: Pyfig, objective: Callable):
	if c.distribute.head:
		study = optuna.create_study(
			study_name		= c.sweep.sweep_name,
			load_if_exists 	= True, 
			direction 		= "minimize",
			storage			= c.sweep.storage,
			sampler 		= lo_ve(c.exp_dir/'sampler.pk') or samplers.TPESampler(seed=c.seed),
			pruner			= pruners.MedianPruner(n_warmup_steps=10),
		)
	else:
		while not c.sweep.storage.exists():
			sleep(3)

	study.optimize(
		objective, 
		n_trials=c.sweep.n_trials, 
		timeout=None, 
		callbacks=None, 
		show_progress_bar=True, 
		gc_after_trial=True
	)

	if c.debug:
		print(vars(study))
		print(study.trials)

	return dict(c_update=study.best_params)

def sync(v_d: torch.Tensor, dist: Accelerator) -> list[torch.Tensor]:
	v_flat, treespec = optree.tree_flatten(v_d)
	v_sync_flat: list[torch.Tensor] =  dist.gather(v_flat)
	for i, (v, v_ref) in enumerate(zip(v_sync_flat, v_flat)):
		v = v.reshape(-1, *v_ref.shape).mean(0)
		if v.is_leaf:
			v.requires_grad = False
		v_sync_mean = [v] if i==0 else [*v_sync_mean, v]
	v_sync = optree.tree_unflatten(treespec=treespec, leaves=v_sync_mean)
	return v_sync

def gen_profile(fn: Callable, wait=1, warmup=1, active=1, repeat=1, v_init: dict=None, c_update: dict=None):
	
	fn = partial(fn, r=v_init.get('r'), deltar=v_init.get('deltar'), mode=v_init.get('mode'), c_update=c_update or dict())

	profiler = torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
		schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
		on_trace_ready=torch.profiler.tensorboard_trace_handler(c.profile_dir),
		profile_memory=True, with_stack=True, with_modules=True
	)
	with profiler:
		for _ in range((wait + warmup + active) * repeat):
			fn()
			profiler.step()

	profiler.export_stacks(c.profile_dir/'profiler_stacks.txt', 'self_cuda_time_total')

	print(profiler.key_averages().table())

	profile_art = wandb.Artifact(f"trace", type="profile")
	p = next(c.profile_dir.iterdir())
	profile_art.add_file(p, "trace.pt.trace.json")
	profile_art.save()

def get_opt(c: Pyfig) -> type:
	if c.opt.opt_name == 'RAdam':
		return partial(torch.optim.RAdam, lr=c.opt.lr)

	if c.opt.opt_name == 'Adahessian':
		import torch_optimizer  # pip install torch_optimizer
		return partial(torch_optimizer.Adahessian,
					lr 			= c.opt.lr,
					betas		= c.opt.betas,
					eps			= c.opt.eps,
					weight_decay= c.opt.weight_decay,
					hessian_power= c.opt.hessian_power)
	raise NotImplementedError

# if c.model.compile_ts:
# 		# os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
# 		model = torch.jit.script(model, r.clone())
		
# 	if c.model.compile_func:
# 		pass
# 		# https://pytorch.org/tutorials/intermediate/nvfuser_intro_tutorial.html
  
# 	if c.model.optimise_ts:
# 		pass
# 		# from torch.utils.mobile_optimizer import optimize_for_mobile
# 		# optimized_torchscript_model = optimize_for_mobile(torchscript_model)
# 		# The optimized model can then be saved and deployed in mobile apps:
# 		# optimized_torchscript_model.save("optimized_torchscript_model.pth")

# 	if c.model.optimise_aot:
# 		# https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html
# 		pass

# 	if c.model.functional:
# 		pass

def run(c: Pyfig=None):
	
	print('\n pyfig \n')
	c = c or Pyfig()
	pprint.pprint(c.d)

	# torch.backends.cudnn.benchmark = True
	# torch.manual_seed(c.seed)
	set_seed(c.seed)
	torch.set_default_tensor_type(torch.DoubleTensor)   # docs:todo ensure works when default not set AND can go float32 or 64
	print(f'{torch.cuda.is_available()*"CUDA and "} 🤖 {c.resource.n_device} GPUs available.')

	c.to(framework='torch')
	dist=dist=Accelerator()
	dtype = c.set_dtype()
	device = c.set_device(device=dist.device)

	### init things ###
	center_points = get_center_points(c.data.n_e, c.data.a)
	r = init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
	deltar_0 = torch.tensor([0.02], device=device, dtype=dtype)

	from hwat import Ansatz_fb
	model: nn.Module = c.partial(Ansatz_fb).to(device=device, dtype=dtype)
	model.requires_grad_(False)
	model_og = deepcopy(model)

	model_fn, params, buffers = make_functional_with_buffers(model_og)  
	model_rv = lambda params, _r: vmap(model_fn, in_dims=(None, None, 0))(params, buffers, _r).sum()
 
	print(f"""exp/actual | 
		cps    : {(c.data.n_e, 3)}/{center_points.shape}
		r      : {(c.resource.n_device, c.data.n_b, c.data.n_e, 3)}/{r.shape}
		deltar : {(c.resource.n_device, 1)}/{deltar_0.shape}
	""")

	### train ###
	from hwat import keep_around_points, sample_b
	from hwat import compute_ke_b, compute_pe_b
	from utils import compute_metrix

	setup_opt = get_opt(c)
	opt = setup_opt(model.parameters())
	opt: torch.optim.Optimizer

	# model, opt = c.setup_distribute(model, opt)
	model, opt = dist.prepare(model, opt)  # docs:accelerate
	model: torch.nn.Module

	def compute_energy(model, r):
		ke = compute_ke_b(model, model_rv, r, ke_method=c.model.ke_method)
		pe = compute_pe_b(r, c.data.a, c.data.a_z)
		e = pe + ke
		return dict(e=e, pe=pe, ke=ke)

	def update_model(model: nn.Module, grads: dict=None, params: dict=None):

		with torch.no_grad():
			if grads is not None:
				for p, g in zip(model.parameters(), grads.values()):
					p.grad = g

			if params is not None:
				for p, p_new in zip(model.parameters(), params.values()):
					p = p_new

			opt.step()
		opt.zero_grad(set_to_none=True)

	def execute(*, r: torch.Tensor=None, deltar: torch.Tensor=None, mode: str='train', c_update: dict=None, **kw):
	 
		c.update(c_update or {})
		
		if mode=='train':
			model.train()
		elif mode=='evaluate':
			model.eval()

		v_d = dict(r=r, deltar=deltar)
  
		if c.debug:
			pprint.pprint(c.d)
			[print(k, v.shape) for k,v in v_d.items()]

		[setattr(p, 'requires_grad', False) for p in model.parameters()]
		opt.zero_grad(set_to_none=True)
  
		for step in range(1, (c.n_step if mode=='train' else c.n_step_eval) + 1):

			def loss_fn(r: torch.Tensor=None, deltar: torch.Tensor=None, **kw):
				
				v_sam = sample_b(model, r, deltar, n_corr=c.data.n_corr)
				r, deltar = v_sam['r'], v_sam['deltar']
				r = keep_around_points(r, center_points, l=5.) if step < c.n_pretrain_step else r
		
				v_e = compute_energy(model, r)

				loss = grads = params = None

				if mode=='train':
					e = v_e['e']
					e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
					e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

					opt.zero_grad()
					for p in model.parameters():
						p.requires_grad = True

					loss: torch.Tensor = ((e_clip - e_clip.mean())*model(r)).mean()

					dist.backward(loss)
	 
					params = {k:p.data for k,p in model.named_parameters()}
					grads = {k:p.grad.data for k,p in model.named_parameters()}

				return dict(loss=loss, grads=grads, params=params, **v_sam, **v_e)
	
			v_d = loss_fn(**v_d)

			if c.distribute.sync_step and (c.resource.n_gpu > 1):
				v_d = sync(v_d, dist)
	
			if mode=='train':
				update_model(model, grads= v_d.get('grads'))
	
			if not (step % c.log_metric_step) :
				if int(c.distribute.rank)==0:
					metrix = compute_metrix(v_d, mode=mode)
					wandb.log(metrix, step=step)
		return v_d

	v_init_0 = v_init = dict(r= r, deltar= deltar_0)

	# 	$ mysql -u root -e "CREATE DATABASE IF NOT EXISTS example"
	# 	$ optuna create-study --study-name "distributed-example" --storage "mysql://root@localhost/example"
	#	[I 2020-07-21 13:43:39,642] A new study created with name: distributed-example

	res = dict(v_init_0=v_init_0, v_init=v_init_0)

	for mode in (c.mode or c.multimode.split(':')):
		v_init = res.get('v_init')
		c.mode = mode

		if mode == 'opt_hypam':

			def objective(trial: Trial):
				c_update = get_hypam_from_study(trial, c.sweep.parameters)
				v_tr = execute(**v_init_0, c_update=c_update)
				v_eval = execute(**v_tr, c_update=c_update)
				return v_eval['e']

			v_run = opt_hypam(c, objective)

		elif mode=='max_mem':
			v_init.update(dict(mode='train'))
			v_run = get_max_mem_c(execute, **res.get('v_init'))

		elif mode=='profile':
			v_init.update(dict(mode='train'))
			v_run = gen_profile(execute, **res.get('v_init'))

		else:
			v_run = execute(**res.get('v_init'), mode=mode)

		c.update(v_run.get('c_update', {}))
		res.update(v_run)
		res[mode] = v_run

	return res


if __name__ == "__main__":
	
	from pyfig import Pyfig 

	c = Pyfig(notebook=False, sweep=None, c_init=None)

	res = run(c)



"""


def gen_profile(
	model: nn.Module, 
	model_rv: nn.Module, 
	r: torch.Tensor=None, 
	deltar: torch.Tensor=None, 
	wait=1, warmup=1, active=1, repeat=1
):
	# https://wandb.ai/wandb/trace/reports/Using-the-PyTorch-Profiler-with-W-B--Vmlldzo5MDE3NjU
	print('profiling')
	model_tmp = deepcopy(model)
	model_pr = lambda : model_tmp(r)
	sample_pr = lambda : sample_b(model_tmp, r, deltar, n_corr=c.data.n_corr) 
	ke_pr = lambda : compute_ke_b(model_tmp, model_rv, r, ke_method=c.model.ke_method)

	profiler = torch.profiler.profile(
		schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
		on_trace_ready=torch.profiler.tensorboard_trace_handler(c.profile_dir),
		profile_memory=True, with_stack=True, with_modules=True
	)
	with profiler:
		times = dict(t_model=0.0, t_sample=0.0, t_ke=0.0)
		import time
		for _ in range((wait + warmup + active) * repeat):
			t0 = time.time()
			model_pr()
			times['t_model'] += time.time() - t0
			t0 = time.time()
			sample_pr()
			times['t_sample'] += time.time() - t0
			t0 = time.time()
			ke_pr()
			times['t_ke'] += time.time() - t0
			profiler.step()
	for k,v in times.items():
		c.wb.run.summary[k] = v

	profile_art = wandb.Artifact(f"trace", type="profile")
	for p in c.profile_dir.iterdir():
		profile_art.add_file(p, "trace.pt.trace.json")
		break
	profile_art.save()
	profiler.export_stacks(c.profile_dir/'profiler_stacks.txt', 'self_cuda_time_total')
	# if not c.model.compile_ts: # https://github.com/pytorch/pytorch/issues/76791
	# docs:profiler
	1- --> wandb --> Artifacts --> files --> trace
	https://wandb.ai/wandb/trace/reports/Using-the-PyTorch-Profiler-with-W-B--Vmlldzo5MDE3NjU
	2- tensorboard --logdir=c.profile_dir
	browser: http://localhost:6006/pytorch_profiler
	https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

	print('profile end.')
 
 """