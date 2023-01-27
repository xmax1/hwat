
from walle.pyfig_utils import Metrix

from typing import Callable

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from functorch import make_functional_with_buffers, vmap
from walle.torch_utils import get_opt, get_scheduler, load, get_max_mem_c
from walle.utils import debug_dict, npify_tree, compute_metrix, flat_any

from walle.pyfig_utils import Param, lo_ve
from pyfig import Pyfig 
from functools import partial

from hwat import Ansatz_fb as Model
from hwat import get_center_points
from hwat import keep_around_points, sample_b
from hwat import compute_ke_b, compute_pe_b
import numpy as np
from copy import deepcopy
import pprint


def init_exp(c: Pyfig, c_update: dict=None, **kw):
	
	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()

	c.update((c_update or {}) | (kw or {}))

	c.set_seed()
	c.set_dtype()
	c.set_device()
	c.to(framework='torch')

	torch.backends.cudnn.benchmark = c.cudnn_benchmark

	model: torch.nn.Module = c.partial(Model).to(device=c.device, dtype=c.dtype)
	model_fn, param, b = make_functional_with_buffers(model)
	model_fn_vmap = vmap(model_fn, in_dims=(None, None, 0))

	opt = get_opt(**c.opt.d_flat)(model.parameters())
	scheduler = get_scheduler(n_step=c.n_step, **c.opt.scheduler.d_flat)(opt)

	if c.lo_ve_path:
		print('init: loading config, modelm and opt \n', )
		c, model, opt = load(c, path=c.lo_ve_path, things_to_load=dict(model=model, opt=opt))
		pprint.pprint(c.d)

	model, opt, scheduler = c.dist.prepare(model, opt, scheduler)

	if 'train' in c.mode:
		model.train()
	elif c.mode=='eval':
		model.eval()

	compute_loss = partial(loss_fn, mode=c.mode, model_fn=model_fn_vmap, model=model)

	return compute_loss, model, opt, scheduler


def loss_fn(
	step: int,
	r: torch.Tensor=None,
	deltar: torch.Tensor=None,
	model:torch.nn.Module=None,
	model_fn: Callable=None,
	mode: str = 'train', 
	**kw, 
):

	with torch.no_grad():

		if step < 0: # c.n_pre_step:
			r = center_points + 10*torch.randn_like(r)

		min_step = max(4, 2*round( (c.n_pre_step * (step/c.n_pre_step)) /2.))
		n_corr = int(min(min_step, c.data.n_corr))
		center_points = get_center_points(c.data.n_e, c.data.a)
		r, acc, deltar = sample_b(model, r, deltar, n_corr=n_corr)

		r = keep_around_points(r, center_points, l=5.+10.*step**1.02/c.n_pre_step)
		pe = compute_pe_b(r, c.data.a, c.data.a_z) 
	
	ke = compute_ke_b(model, model_fn, r, ke_method=c.model.ke_method)

	with torch.no_grad():
		e = pe + ke
		e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
		e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)
		energy = (e_clip - e_clip.mean())
		
	loss, grads, params = None, None, None

	if 'train' in mode:
		loss = ((energy / c.data.a_z.sum()) * model(r)).mean() 

	v_init = dict(r=r, deltar=deltar)

	return loss, v_init, dict(r=r, deltar=deltar, e=e, pe=pe, ke=ke, acc=acc, loss=loss)

import wandb

def run(c: Pyfig=None, c_update: dict=None, v_init: dict=None, **kw):

	compute_loss, model, opt, scheduler = init_exp(c, c_update, **kw)
	
	v_init = c.init_app(v_init)

	metrix = Metrix(eval_keys=c.eval_keys)
	
	c.start(dark='dark' in c.mode)

	c.log_metric_step = min(c.log_metric_step, c.n_step//2)

	v_cpu_d = {}

	print('running: ', c.mode, 'run: \n', compute_loss, 'c_update: \n', c_update)
	for rel_step, step in enumerate(range(1, (c.n_step if 'train' in c.mode else c.n_eval_step) + 1)):

		model.zero_grad(set_to_none=True)

		loss, v_init, v_d = compute_loss(step, **v_init)

		if 'train' in c.mode:
			c.dist.backward(loss)
			v_d['grads'] = {k:p.grad.detach() for k,p in model.named_parameters()}
		
		if (not (step % c.dist.sync_step)) and (c.resource.n_gpu > 1):
			v_d = c.dist.sync(step, v_d)


		if 'train' in c.mode:
			with torch.no_grad():
				for k, p in model.named_parameters():
					g = v_d.get('grads').get(k) * (0.01 * step < c.n_pre_step)
					p.grad.copy_( g )
				opt.step()
				scheduler.step()


		###--- cpu only from here ---###
		if ('eval' in c.mode) or not (step % c.log_metric_step):

			v_d['grads'] = {k:p.grad.detach().cpu().numpy() for k,p in model.named_parameters() if not p.grad is None}
			v_d['params'] = {k:p.detach().cpu().numpy() for k,p in model.named_parameters()}

			if not 'debug' in c.mode:
				v_d.pop('grads')
				v_d.pop('params')

			if 'eval' in c.mode:
				keep_keys = c.eval_keys + list(v_init.keys()) + c.log_keys
				v_d = dict(filter(lambda kv: kv[0] in keep_keys, v_d.items()))

			v_cpu_d = npify_tree(v_d)

			v_cpu_d |= metrix.tick(step, opt_obj=v_cpu_d['e'].mean())

			v_metrix = compute_metrix(v_cpu_d, source=c.mode, sep='/')

			if int(c.dist.rank)==0 and c.dist.head:

				if not 'dark' in c.mode:
					wandb.log(v_metrix, step=step)

				if 'record' in c.mode:
					log_data = dict(filter(lambda kv: kv[0] in c.log_keys, v_d.items()))
					log_data = {k:v[None] for k,v in log_data.items()}
					metrix.log = log_data if rel_step==1 else {k:np.stack(v, log_data[k]) for k,v in log_data.items()}

		if int(c.dist.rank)==0 and c.dist.head:
			if not (step % c.log_state_step) and '-record' in c.mode:
				lo_ve(path=c.state_dir/f'{c.mode}_i{step}.state', data=v_d)


	v_run = dict(v_init_next={k:v.detach().cpu().numpy() for k,v in v_init.items()})

	c.to(framework='numpy')

	torch.cuda.empty_cache()

	if int(c.dist.rank)==0 and c.dist.head:

		if not 'dark' in c.mode:
			n_param = sum(p.numel() for p in model.parameters())
			wandb.log(dict(n_param=n_param))
			done = c.record_app(metrix.opt_obj_all)

		if metrix.log:
			lo_ve(path=(c.exp_dir/c.run_id).with_suffix('.npz'), data=metrix.log)  # ! mean gather used by core loop, need sth else

	if v_run and v_cpu_d and metrix.to_dict():
		v_run |= v_cpu_d | metrix.to_dict()

	c.end()

	if v_init.get('next', '')=='eval':
		# v_init['next_state_path'] = c.state_dir/f'{c.mode}_i{step}.state'
		path = c.state_dir/f'{c.mode}_i{step}.state'
		lo_ve(path=path, data=v_d)
		v_run['c_update_next'] = {'lo_ve_path': path} 

	return v_run


if __name__ == "__main__":
	
	allowed_mode_all = 'train:eval:max_mem:opt_hypam:profile:train-eval'

	c = Pyfig(notebook=False, sweep=None, c_update=None)

	class RunMode:

		def __call__(ii, c: Pyfig, v_init: dict=None, c_update: dict=None, mode: str=None):
			c.update(c_update | dict(mode=(mode or {})))
			next_mode = v_init.get('next', 'evaluate')
			fn = getattr(ii, c.mode.split('-')[0])

			return fn(c=c, v_init=v_init)

		def opt_hypam(ii, c: Pyfig=None, v_init: dict=None):
			from walle.opt_hypam_utils import opt_hypam, objective
   
			objective = partial(objective, c=c, run=run)
			v_run = opt_hypam(objective, c)

			#
			# Create the summary run.
			# summary = wandb.init(project="optuna",
			# 					name="summary",
			# 					job_type="logging")

			# Getting the study trials.
			# trials = study.trials

			# WandB summary.
			# for step, trial in enumerate(trials):
			# 	# Logging the loss.
			# 	summary.log({"mse": trial.value}, step=step)

			# 	# Logging the parameters.
			# 	for k, v in trial.params.items():
			# 		summary.log({k: v}, step=step)


			return v_run

		def max_mem(ii, c: Pyfig=None, v_init: dict=None):

			from walle.torch_utils import get_max_mem_c

			v_run = get_max_mem_c(run, c=c)
			# dict(n_b_max=n_b, max_mem_alloc=mem_used)
			# now = datetime.now().strftime("%d-%m-%y:%H-%M-%S")
			# line = now + ',' + ','.join([str(v) for v in 
			# 	[v_run['max_mem_alloc'], c.data.n_b, c.data.n_e, c.model.n_fb, c.model.n_sv, c.model.n_pv]])
			# with open('./dump/max_mem.csv', 'a+') as f:
			# 	f.writelines([line])
			v_run['v_init_next'] = v_run
			return v_run

		def train(ii, c: Pyfig=None, v_init: dict=None):
			return run(c=c, v_init=v_init)
			
		def eval(ii, c: Pyfig=None, v_init: dict=None):
			return run(c=c, v_init=v_init)

		def profile(ii, c: Pyfig=None, v_init: dict=None):
			from walle.torch_utils import gen_profile
			c.mode = 'train'
			fn = partial(run, c=c, v_init=v_init)
			v_run = gen_profile(fn, c)
			return v_run

	run_mode = RunMode()

	res, v_run = dict(), dict(v_init_next=dict())

	run_mode_all = [c.mode,] or c.multimode.split(':')
	print('run.py:mode = \n ***', run_mode_all, '***')
	for mode_i, mode in enumerate(run_mode_all):

		next_i = mode_i+1
		if next_i < len(run_mode_all):
			v_run['v_init_next']['next'] = run_mode_all[next_i]

		c_update_next = v_run.get('c_update_next', {})

		v_init_next = v_run.get('v_init_next', {})

		c.mode = mode

		v_run = run_mode(c, v_init=v_init_next, c_update=c_update_next)

		res[mode] = deepcopy(v_run)

	
