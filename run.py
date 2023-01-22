import os
import torch
import time
import wandb
from pyfig import Pyfig

from optuna import Trial
import sys

from torch.optim import Optimizer
from torch import nn
from functorch import make_functional_with_buffers, vmap
from torch_utils import update_model
from utils import compute_metrix
from torch_utils import get_opt, get_scheduler, load
from utils import get_max_n_from_filename, debug_dict, numpify_tree, torchify_tree, cpuify_tree, lo_ve
from pyfig_utils import Param

from hwat import Ansatz_fb as Model
from hwat import init_r, get_center_points
from hwat import keep_around_points, sample_b
from hwat import compute_ke_b, compute_pe_b


def init_exp(c: Pyfig, c_init: dict=None, **kw):
	c_init = (c_init or {}) | (kw or {})
	c.update(c_init)

	c.set_dtype()
	c.distribute.set_seed()
	c.distribute.set_device()
	c.to(device=c.distribute.device, dtype=c.dtype)

	model: nn.Module = c.partial(Model).to(device=c.distribute.device, dtype=c.dtype)
	model_fn, params, buffers = make_functional_with_buffers(model)  
	model_fn_vmap = lambda params, *_v: vmap(model_fn, in_dims=(None, None, 0))(params, buffers, *_v).sum()
	
	opt = get_opt(**c.opt.d_flat)(model.parameters())
	scheduler = c.partial(get_scheduler)(opt)

	if c.lo_ve_path:
		c, model, opt, r = load(c, things_to_load=dict(model=model, opt=opt, r=r))

	model, opt, scheduler, buffers = c.distribute.prepare(model, opt, scheduler, buffers, )

	if c.mode=='train':
		model.train()
	elif c.mode=='evaluate':
		model.eval()

	return dict(model=model, model_fn_vmap=model_fn_vmap, opt=opt, scheduler=scheduler)


def run(c: Pyfig=None, c_init: dict=None, **kw):

	c = c or Pyfig()
	print('torch says: count-', torch.cuda.device_count(), 'device-', torch.cuda.current_device())
	print(f'{torch.cuda.is_available()*"CUDA and "} 🤖 {c.resource.n_device} GPUs available. {os.environ["CUDA_VISIBLE_DEVICES"]}')
	
	things = init_exp(c, c_init, **kw)

	def get_app_things(ii):
		center_points   = get_center_points(c.data.n_e, c.data.a)
		r_0 			= init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
		deltar		= torch.tensor([0.02,], device=c.distribute.device, dtype=c.dtype)
		return dict(r=r_0, deltar=deltar) 

	app_things = get_app_things(c)

	c.start()
	
	from typing import Callable
	from torch.optim.lr_scheduler import _LRScheduler

	def loop(
		c: Pyfig,
		v_d: dict				= None,
		model: 	nn.Module 		= None,
		model_fn_vmap: Callable	= None,
		opt: Optimizer			= None,
		scheduler: _LRScheduler = None
	):

		def loss_fn(step: int, v_d: dict):
			
			opt.zero_grad(set_to_none=True)
			[setattr(p, 'requires_grad', False) for p in model.parameters()]
			params = [p.detach().data for p in model.parameters()]
			model_rv = lambda _r: model_fn_vmap(params, _r)

			###--- start app ---###
			with torch.no_grad():
				v_app_d = sample_b(model, v_d['r'], v_d['deltar'], n_corr=c.data.n_corr)

				if step < c.n_pre_step:
					r = keep_around_points(v_app_d['r'], c.data.center_points, l=5.+10.*step/c.n_pre_step)

				pe = compute_pe_b(r, c.data.a, c.data.a_z)

			ke = compute_ke_b(model, model_rv, v_app_d['r'], ke_method=c.model.ke_method)

			v_app_d = v_app_d | dict(e=pe+ke, pe=pe, ke=ke, opt_obj=e.mean())
			###--- end app ---###

			loss = grads = params = None

			if c.mode=='train':
				opt.zero_grad(set_to_none=True)
				for p in model.parameters():
					p.requires_grad = True

				###--- start app loss ---###
				e = v_app_d['e']
				e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
				e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)
				loss: torch.Tensor = ((e_clip - e_clip.mean())*model(r)).mean()
				###--- end app loss ---###

				c.distribute.backward(loss)

			return dict(loss=loss, grads=grads, params=params) | v_app_d

				params = {k:p.data for k,p in model.named_parameters()}
				grads = {k:p.grad.data for k,p in model.named_parameters()}

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
			v_d = {k:v.detach() for k,v in v_d.items() if k in v_d['loop_vars']}
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

	return res


if __name__ == "__main__":
	
	from pyfig import Pyfig 

	c = Pyfig(notebook=False, sweep=None, c_init=None)

	res = run(c)
 
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
					v_tr = run(c=c, init_d=c_update)
					c.mode = 'evaluate'
					v_eval = run(c=c, **v_tr)
					return torch.stack(v_eval['opt_obj_all']).mean()

				v_run = opt_hypam(objective, c)

				debug_dict(v_run, msg='opt_hypam:v_run')
				
				sys.exit('now figure out cnversion')

			elif mode=='max_mem':
				from torch_utils import get_max_mem_c
				from datetime import datetime
				v_run = get_max_mem_c(run, mode='train', n_step=2*c.log_metric_step)
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
				v_run = gen_profile(run, profile_dir=c.profile_dir, mode='train', **v_run)
			else:
				print('train or evaluate')
				v_run = run(c=c, **v_run)

			c.update(**v_run)
			res[mode] = v_run

		except Exception as e:
			tb = traceback.format_exc()
			print(f'mode loop error={e}')
			debug_dict(msg='c.d=', d=c.d_flat)
			debug_dict(msg='v_run=', d=v_run)
			print(f'traceback={tb}')
