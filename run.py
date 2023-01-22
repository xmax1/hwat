import os
import torch
import time
import numpy as np
from utils import flat_any
from pathlib import Path
from time import sleep
import shutil
from functools import partial
from copy import deepcopy
from pyfig import Pyfig

from copy import deepcopy

import optuna
from optuna import pruners, samplers, Trial
import sys

from torch.optim import Optimizer
from torch import nn
from functorch import make_functional_with_buffers, vmap
from torch_utils import update_model
from utils import compute_metrix
from torch_utils import get_opt, get_scheduler
from utils import get_max_n_from_filename, debug_dict, numpify_tree, torchify_tree, cpuify_tree, lo_ve
from pyfig_utils import Param

from hwat import Ansatz_fb as Model
from hwat import init_r, get_center_points
from hwat import keep_around_points, sample_b
from hwat import compute_ke_b, compute_pe_b


def get_app_things(c: Pyfig):
	center_points   = get_center_points(c.data.n_e, c.data.a)
	r_0 			= init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
	deltar_0		= torch.tensor([0.02,], device=c.distribute.device, dtype=c.dtype)
	return dict(r=r_0, deltar=deltar_0) 


def run(c: Pyfig=None):

	c = c or Pyfig()
 
	print('torch says: count-', torch.cuda.device_count(), 'device-', torch.cuda.current_device())
	print(f'{torch.cuda.is_available()*"CUDA and "} 🤖 {c.resource.n_device} GPUs available. {os.environ["CUDA_VISIBLE_DEVICES"]}')

	def execute(
		c: 		Pyfig			= c,
		c_init: dict			= None,
	):

		things = get_things(c, c_init=c_init)

		app_things = get_app_things(c)

		def loss_fn(step:int, *, r: torch.Tensor=None, deltar: torch.Tensor=None, **__kw):
	  
			opt.zero_grad(set_to_none=True)
			[setattr(p, 'requires_grad', False) for p in model.parameters()]
			params = [p.detach().data for p in model.parameters()]
			model_rv = lambda _r: model_rv_0(params, _r)

			###--- start app ---###
			with torch.no_grad():
				v_app_d = sample_b(model, r, deltar, n_corr=c.data.n_corr)
				r, deltar = v_app_d['r'], v_app_d['deltar']

				if step < c.n_pre_step:
					center_points = get_center_points(c.data.n_e, c.data.a)
					r = keep_around_points(r, center_points, l=5.+10.*step/c.n_pre_step)

				pe = compute_pe_b(r, c.data.a, c.data.a_z)

			ke = compute_ke_b(model, model_rv, r, ke_method=c.model.ke_method)

			e = pe + ke
			v_app_d = v_app_d | dict(e=e, pe=pe, ke=ke, opt_obj=e.mean())
			###--- end app ---###

			loss = grads = params = None

			if c.mode=='train':
				opt.zero_grad(set_to_none=True)
				for p in model.parameters():
					p.requires_grad = True

				###--- start app ---###
				e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
				e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

				loss: torch.Tensor = ((e_clip - e_clip.mean())*model(r)).mean()
				###--- end app ---###

				c.distribute.backward(loss)

				params = {k:p.data for k,p in model.named_parameters()}
				grads = {k:p.grad.data for k,p in model.named_parameters()}

			v_d = dict(loss=loss, grads=grads, params=params) 

			return v_d | v_app_d

		
		def loop(
			c: Pyfig,
			v_d: dict				= None
			model: 	nn.Module 		= None,
			model_fn_vmap: Callable	= None,
			buffers: dict			= None,
			opt: Optimizer			= None
   		):  

			t0 = time.time()
			torch.cuda.reset_peak_memory_stats()

			for step in range(1, (c.n_step if c.mode=='train' else c.n_eval_step) + 1):

				v_d: dict = loss_fn(step, **v_d)
				debug_dict(msg='loss_fn:v_d', d=v_d, step=step)

				if (not (step % c.distribute.sync_step)):
					if c.resource.n_gpu > 1:
						v_d = c.distribute.sync(step, v_d)

				if c.mode=='train':
					update_model(model, grads= v_d.get('grads'), step=step)
					scheduler.step()
					opt.step()

				v_cpu_d = cpuify_tree(v_d)
				v_d = {k:v.detach() for k,v in v_d.items() if k in do_not_clear}
				###--- cpu only from here ---###

				if not (step % c.log_metric_step):
					if int(c.distribute.rank)==0:
						metrix = compute_metrix(v_cpu_d, mode=c.mode)
						debug_dict(msg='metrix', metrix=metrix, step=step//c.log_metric_step)
						wandb.log(metrix, step=step)
						# c.wb.run.summary[k] = v.abs().mean()
	  
					v_cpu_d['max_mem_alloc'] = torch.cuda.max_memory_allocated() // 1024 // 1024
					torch.cuda.reset_peak_memory_stats() 

					t_diff, t0 = time.time() - t0, time.time()
					v_cpu_d['t_per_it'] = t_diff/c.distribute.sync_step
		
				v_cpu_d['opt_obj_all'] = v_cpu_d.get('opt_obj_all', []) + [v_cpu_d['opt_obj'],]

			torch.cuda.empty_cache()

			return numpify_tree(v_cpu_d) # end loop
		v_d_numpy = loop(v_init_d)
		return v_d_numpy # end execute

	###--- start run ---###
	if c.lo_ve_path:
		c, model, opt, r = load(c, things_to_load=dict(model=model, opt=opt, r=r))
 
	import traceback
	v_run = dict()
	res = dict()
	for mode in ([c.mode,] if c.mode else c.multimode.split(':')):
		print('Running: ', mode)
		c.mode = mode

		try:
			if mode == 'opt_hypam':

				from opt_hypam_utils import get_hypam_from_study, opt_hypam

				def objective(trial: Trial):
					c_update = get_hypam_from_study(trial, c.sweep.parameters)
					c.mode = 'train'
					v_tr = execute(c=c, init_d=c_update)
					c.mode = 'evaluate'
					v_eval = execute(c=c, **v_tr)
					return torch.stack(v_eval['opt_obj_all']).mean()

				v_run = opt_hypam(objective, c)

				debug_dict(v_run, msg='opt_hypam:v_run')
				
				sys.exit('now figure out cnversion')

			elif mode=='max_mem':
				from torch_utils import get_max_mem_c
				from datetime import datetime
				v_run = get_max_mem_c(execute, mode='train', n_step=2*c.log_metric_step)
				if c.save:
					now = datetime.now().strftime("%d-%m-%y:%H-%M-%S"), 
					line = now + ',' + ','.join([str(v) for v in 
								  [v_run['max_mem_alloc'], c.data.n_b, c.data.n_e, c.model.n_fb, c.model.n_sv, c.model.n_pv]])
					with open('./dump/mem.csv', 'a+') as f:
						f.writelines([line])
	
						
					
			elif mode=='profile':
				pass
				from torch_utils import gen_profile
				debug_dict(d=v_run, msg='profile:init_d')
				v_run = gen_profile(execute, profile_dir=c.profile_dir, mode='train', **v_run)
			else:
				print('train or evaluate')
				v_run = execute(c=c, **v_run)

			c.update(**v_run)
			res[mode] = v_run

		except Exception as e:
			tb = traceback.format_exc()
			print(f'mode loop error={e}')
			debug_dict(msg='c.d=', d=c.d_flat)
			debug_dict(msg='v_run=', d=v_run)
			print(f'traceback={tb}')

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