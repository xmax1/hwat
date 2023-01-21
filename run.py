import os
from accelerate import Accelerator
import torch
from typing import Callable
import time
import accelerate
import numpy as np
from utils import flat_any
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
	return c_update

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
			print('waiting for opt storage...')
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

def update_model(
    model: nn.Module, 
    grads: dict[str:torch.Tensor]=None, 
    params: dict[str:torch.Tensor]=None,
    step: int = 0
):
	try:
		with torch.no_grad():
			for k, p in model.named_parameters():
				if params is not None:
					p.data: torch.Tensor = params[k]
					if step==1:
						[print(k, v.abs().mean(), v.abs().std()) for k,v in params.items()]
				if grads is not None:
					p.grad.data = grads[k]
					if step==1:
						[print(k, v.abs().mean(), v.abs().std()) for k,v in grads.items()]

	except:
		print('model, params, grads', len(grads or {}), len(params or {}), len(list(model.named_parameters())))
		grads = grads or dict(model.named_parameters())
		params = params or dict(model.named_parameters())
		for (k,v), (k_p,v_p), (k_g,v_g) in zip(model.named_parameters(), params.items(), grads.items()):
			print(k, v.shape, k_p, v_p.shape, k_g, v_g.shape)
		sys.exit('shapes')
  
	# def get_opt(c) -> type:
	# 	if c.opt.opt_name == 'RAdam':
	# 		return partial(torch.optim.RAdam, lr=c.opt.lr)

	# 	if c.opt.opt_name == 'Adahessian':
	# 		import torch_optimizer  # pip install torch_optimizer
	# 		return partial(torch_optimizer.Adahessian,
	# 					lr 			= c.opt.lr,
	# 					betas		= c.opt.betas,
	# 					eps			= c.opt.eps,
	# 					weight_decay= c.opt.weight_decay,
	# 					hessian_power= c.opt.hessian_power)
	# 	raise NotImplementedError
	# # from torch_utils import get_opt
	# # setup_opt = get_opt(c)
	# # opt = setup_opt(model.parameters())

def run(c: Pyfig=None):

	c = c or Pyfig()

	from hwat import Ansatz_fb
	from hwat import keep_around_points, sample_b
	from hwat import compute_ke_b, compute_pe_b
	from utils import compute_metrix
 
	c.to(framework='torch')
	dtype = c.set_dtype()
	c.distribute.set_seed()
	device = c.distribute.set_device()

	print('torch says: count-', torch.cuda.device_count(), 'device-', torch.cuda.current_device(), )
	print(f'{torch.cuda.is_available()*"CUDA and "} 🤖 {c.resource.n_device} GPUs available. {os.environ["CUDA_VISIBLE_DEVICES"]}')

	### init things ###
	center_points = get_center_points(c.data.n_e, c.data.a)
 
	def execute(*, 
        c: Pyfig=c,
        **kw
	):

		print('\n ***** Starting Run ***** ')
		c.start((kw or {}))

		r 		=	r 		or init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
		deltar 	= 	deltar 	or torch.tensor([0.02,], device=device, dtype=dtype)
	
		model: nn.Module = c.partial(Ansatz_fb).to(device=device, dtype=dtype)
		model.requires_grad_(False)
		model_og = deepcopy(model)

		model_fn, params, buffers = make_functional_with_buffers(model_og)  
		model_rv_0 = lambda params, _r: vmap(model_fn, in_dims=(None, None, 0))(params, buffers, _r).sum()
  
		opt = torch.optim.RAdam(model.parameters(), lr=c.opt.lr)
		scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c.opt.max_lr, steps_per_epoch=c.n_step, epochs=1)
		# model, opt = c.distribute.prepare(model, opt)  # docs:accelerate
		model: torch.nn.Module
		opt: torch.optim.Optimizer
  
		if c.mode=='train':
			model.train()
		elif c.mode=='evaluate':
			model.eval()

		if c.debug:
			print('execute pyfig and init variables')
			pprint.pprint(c.d)
			print(r.shape, deltar.shape)
  
		### loss fn definition ###
		def loss_fn(step:int, *, r: torch.Tensor=None, deltar: torch.Tensor=None, **kw):
				
			opt.zero_grad(set_to_none=True)
			[setattr(p, 'requires_grad', False) for p in model.parameters()]
			params = [p.detach().data for p in model.parameters()]
			model_rv = lambda _r: model_rv_0(params, _r)

			v_sam = sample_b(model, r, deltar, n_corr=c.data.n_corr)
			r, deltar = v_sam['r'], v_sam['deltar']

			if step < c.n_pre_step:
				r = keep_around_points(r, center_points, l=5.+10.*step/c.n_pre_step)

			ke = compute_ke_b(model, model_rv, r, ke_method=c.model.ke_method)
			pe = compute_pe_b(r, c.data.a, c.data.a_z)
			e = pe + ke
			v_e = dict(e=e, pe=pe, ke=ke)

			loss = grads = params = None

			if c.mode=='train':
				e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
				e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

				opt.zero_grad(set_to_none=True)
				for p in model.parameters():
					p.requires_grad = True

				loss: torch.Tensor = ((e_clip - e_clip.mean())*model(r)).mean()

				c.distribute.backward(loss)

				params = {k:p.data for k,p in model.named_parameters()}
				grads = {k:p.grad.data for k,p in model.named_parameters()}

			v_d = dict(loss=loss, grads=grads, params=params, **v_sam, **v_e)

			if c.debug and step==1:
				for k,v in flat_any(v_d).items():
					if isinstance(v, torch.Tensor):
						c.wb.run.summary[k] = v.abs().mean()
						print(v.abs().mean(), v.abs().std())

			v_d, treespec = optree.tree_flatten(v_d)
			v_d: list[torch.Tensor]
			v_d = [v.detach() for v in v_d]
			if c.debug and step==1:
				[print(v.shape) for v in v_d]

			return optree.tree_unflatten(treespec=treespec, leaves=v_d)


		### train loop ###

		v_d = dict(r=r, deltar=deltar)

		for step in range(1, (c.n_step if c.mode=='train' else c.n_eval_step) + 1):

			v_d = loss_fn(step, **v_d)

			if (not (step % c.distribute.sync_step)) and (c.resource.n_gpu > 1):
				v_d = c.distribute.sync(step, v_d)

			if c.mode=='train':
				update_model(model, grads= v_d.get('grads'), step=step)
				scheduler.step()
				opt.step()

			if not (step % c.log_metric_step):
				if int(c.distribute.rank)==0:
					metrix = compute_metrix(v_d, mode=c.mode)
					wandb.log(metrix, step=step)

				if ((step//c.log_metric_step)==1) and c.debug:
					pprint.pprint(metrix)
					pprint.pprint(v_d)
					pprint.pprint(c.d)

		return v_d


	res = dict()
	for mode in ([c.mode,] if c.mode else c.multimode.split(':')):
		print('Running: ', mode)
		c.mode = mode

		if mode == 'opt_hypam':
			print('hypam opt')

			def objective(trial: Trial):
				c_update = get_hypam_from_study(trial, c)
				c.update(c_update)
				c.mode = 'train'
				v_tr = execute(c=c)
				c.mode = 'evaluate'
				v_eval = execute(c=c, **v_tr)
				return v_eval['e']

			v_run = opt_hypam(c, objective)

		elif mode=='max_mem':
			print('max mem')
			v_run = get_max_mem_c(partial(execute, c=c, **res.get('v_init', {})))

		elif mode=='profile':
			print('profile')
			from torch_utils import gen_profile
			v_run = gen_profile(partial(execute, c=c, **res.get('v_init', {})))

		else:
			print('train or evaluate')
			v_run = execute(c=c, **res.get('v_init', {}))

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