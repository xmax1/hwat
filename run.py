
import wandb
from collections import OrderedDict

from typing import Callable

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from functorch import make_functional_with_buffers, vmap
from things.for_torch.torch_utils import get_opt, get_scheduler, load
from things.utils import npify_tree, compute_metrix

from functools import partial

from hwat import Ansatz_fb as Model
from hwat import compute_ke_b, compute_pe_b
import numpy as np
from copy import deepcopy
import pprint

from things.pyfig_utils import lo_ve
from things.utils import Metrix
from pyfig import Pyfig 

def init_exp(c: Pyfig, c_update: dict=None, **kw):

	c.update((c_update or {}) | (kw or {}))

	c.set_seed()
	c.set_dtype()
	c.set_device()
	c.to(framework='torch')

	torch.backends.cudnn.benchmark = c.cudnn_benchmark

	model: torch.nn.Module = c.partial(Model).to(dtype=c.dtype)
	model_fn, param, buffer = make_functional_with_buffers(model)
	model_fn_vmap = vmap(model_fn, in_dims=(None, None, 0))

	### under construction ###
	from hwat import PyfigDataset
	from torch.utils.data import DataLoader
	dataset = PyfigDataset(c, model)
	def custom_collate(batch):
		return batch[0] 
	dataloader = DataLoader(dataset, batch_size= 1, collate_fn= custom_collate)  # c.data.n_b otherwise because of the internal sampler
	### under construction ###

	opt = get_opt(**c.opt.d_flat)(model.parameters())
	scheduler = get_scheduler(**c.opt.scheduler.d_flat)(opt)

	if c.lo_ve_path:
		print('init: loading config, modelm and opt \n', )
		c, model, opt = load(c, path=c.lo_ve_path, things_to_load=dict(model= model, opt= opt))
		pprint.pprint(c.d)

	model, dataloader, opt, scheduler = c.dist.prepare(model, dataloader, opt, scheduler)

	if 'train' in c.mode:
		model.train()
	elif c.mode=='eval':
		model.eval()

	compute_loss = partial(loss_fn, mode=c.mode, model_fn=model_fn_vmap, model=model)

	model.to(device=c.device)
	print('run:init: ', next(model.parameters()).device, next(model.parameters()).dtype, model, opt, scheduler, sep='\n')

	return model, dataloader, compute_loss, opt, scheduler


def loss_fn(
	data: torch.Tensor=None,
	model:torch.nn.Module=None,
	model_fn: Callable=None,
	phase: str = 'train',
	**kw, 
):

	with torch.no_grad():
		pe = compute_pe_b(data, c.app.a, c.app.a_z) 
	
	try:
		ke = compute_ke_b(model, model_fn, data, ke_method=c.model.ke_method)
	except Exception as e:
		print('ke error: ', e)
		ke = torch.where(torch.isnan(ke), torch.zeros_like(pe), ke)

	with torch.no_grad():
		e = pe + ke
		e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
		e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)
		energy = (e_clip - e_clip.mean())
		
	loss, grads, params = None, None, None

	if 'train' in c.mode:
		# if the other loss variable is detached, then the gradients are not computed
		# sometimes, the one of the variables doesn't have gradients and you forget

		if phase==c.tag.pre:
			model: Model

			# unwrap_model = c.dist.unwrap(model)
			m_orb_u, m_orb_d = model.compute_hf_orb(data, c.app.mol, c.app.hf)
			m_orb = model.full_det_from_spin_det(m_orb_u, m_orb_d)
			
			data = data.detach().requires_grad_(True)
			orb_u, orb_d = model.compute_orb(data)
			orb = model.full_det_from_spin_det(orb_u, orb_d)
			
			loss = ((orb - m_orb)**2).mean(0).sum()
			
		elif phase==c.tag.train:
			loss = ((energy / c.app.a_z.sum()) * model(data)).mean()

		else:
			raise ValueError('phase not recognised: ', phase)

	return loss, dict(data=data, e=e, pe=pe, ke=ke)

@torch.no_grad()
def update(model, grads, opt, scheduler):
	for k, p in model.named_parameters():
		if not (p.grad is None):
			g = grads.get(k)
			p.grad.copy_(g if g is not None else torch.zeros_like(p))
	opt.step()
	scheduler.step()
		

# set rank = 0 for all processes for opt hypam
# 

def run(c: Pyfig=None, c_update: dict=None, v_init: dict=None, **kw):

	model, dataloader, compute_loss, opt, scheduler = init_exp(c, c_update, **kw)
	
	c.start()
	
	initial_state_metrix = dict(n_param=sum(p.numel() for p in model.parameters()))
	metrix = Metrix(mode=c.mode, init_summary=initial_state_metrix)

	print('depreciating v_init: keys=', (v_init or {}).keys())
	print('running: ', c.mode, 'run: ', compute_loss, 'c_update: ', c_update, sep='\n')

	v_cpu_d = dict()

	subrtne = (
		(c.tag.pre, c.n_pre_step),
		(c.tag.train, c.n_step),
		(c.tag.eval, c.n_eval_step),
	)

	rtne = {
		c.tag.train : (subrtne[0], subrtne[1]),
		c.tag.eval	: (subrtne[2],),
	}

	for phase, n_phase_step in rtne[c.mode.split('-')[0]]: # ! because of the -option
		metrix.phase = phase

		for (rel_step, step), data in zip(enumerate(range(1, n_phase_step+1), 1), dataloader):

			model.zero_grad(set_to_none=True)

			loss, v_d = compute_loss(data, model=model, phase=phase)

			if c.tag.train in c.mode:

				create_graph = c.opt.opt_name.lower()=='AdaHessian'.lower()
				c.dist.backward(loss, create_graph=create_graph)
				no_none = filter(lambda kv: not (kv[1].grad is None), model.named_parameters())
				v_d['grads'] = {k:p.grad.detach() for k,p in no_none}

			v_d = c.dist.sync(v_d, sync_method=c.tag.mean, this_is_noop= not (step % c.dist.sync_step))

			if c.tag.train in c.mode:

				update(model, v_d['grads'], opt, scheduler)

				v_d['grads'] = {k:g.cpu().numpy() for k,g in v_d['grads'].items()}
				v_d['params'] = {k:p.detach().cpu().numpy() for k,p in model.named_parameters()}


			###--- cpu only from here ---###
			### construction zone ###
			log_metric_step = n_phase_step // c.log_metric_n_times
			log_state_step = n_phase_step // c.log_state_n_times
			log_metrix = not (step % log_metric_step)
			log_state = not (step % log_state_step)
			log_final = step==n_phase_step
			is_log_step = log_metrix or log_state or log_final
			### construction zone ###

			if c.dist.head and is_log_step:

				v_cpu_d = npify_tree(v_d)

				if log_metrix:
					v_metrix = metrix.tick(step,
						opt_obj_key= c.opt_obj_key, opt_obj_op= c.opt_obj_op, v_cpu_d= v_cpu_d,
						log_exp_stats_keys=c.log_exp_stats_keys
					)

					wandb.log(compute_metrix(v_metrix, source=c.mode, sep='/'), step=step)

				if log_state:
					lo_ve(path=c.state_dir/f'{c.mode}_{phase}_i{step}.state', data=v_cpu_d)

			if rel_step==1 or rel_step==n_phase_step:
				print(f'{c.mode} {phase} step {step} of {n_phase_step} done')
				v_cpu_d_1 = {k:f'{v.mean():.2f}' for k,v in v_cpu_d.items()}
				other = dict(
					opt_obj_key= c.opt_obj_key, 
					opt_obj_op= c.opt_obj_op,
					step=step, 
					n_step=n_phase_step, 
					log_metric_step=log_metric_step, 
					log_state_step=log_state_step,
				)
				pprint.pprint(v_cpu_d_1 | other)

	c.to(framework='numpy')

	torch.cuda.empty_cache()

	if c.dist.head:
		c.app.record_summary(summary= metrix.summary, opt_obj_all=metrix.opt_obj_all)

		if c.log_data_dump_keys is not None:
			
			v_data_dump: dict = c.dist.sync(v_cpu_d, sync_method=c.tag.gather)

			data_dump = dict(filter(lambda kv: kv[0] in c.log_data_dump_keys, v_data_dump.items()))

			lo_ve(path=(c.exp_dir/c.run_id).with_suffix('.npz'), data=data_dump)


	if v_init.get(c.tag.next_run)==c.tag.eval:
		
		path = c.next_run_state_path

		lo_ve(path=path, data=v_cpu_d)

		v_run[c.tag.next_run_c_update] = dict(lo_ve_init_path= path)

	c.end()

	return v_run


if __name__ == "__main__":
	
	allowed_mode_all = 'train:eval:max_mem:opt_hypam:profile:train-eval'

	c = Pyfig(notebook=False, sweep=None, c_update=None)
	c_d0 = c.d_flat

	class RunMode:

		def __call__(ii, c: Pyfig, v_init: dict=None, c_update: dict=None, mode: str=None):
			try:
				
				c.update(c_update | dict(mode=(mode or {})))

				next_mode = v_init.get('next', 'eval')

				fn = getattr(ii, c.mode.split('-')[0])
				
				v_run = fn(c=c, v_init=v_init)

				if next_mode=='opt_hypam':
					v_run['v_init_next'] = {}
					v_run['c_update_next'] = {}

			except Exception as e:
				import traceback
				print(f'RunMode:fail \n', c.mode, e, v_init, c_update, '\nc=\n')
				print(traceback.format_exc())
				pprint.pprint(c.d)
				v_run = {}

			return v_run

		def opt_hypam(ii, c: Pyfig=None, v_init: dict=None):
			from things.opt_hypam_utils import opt_hypam, objective
   
			def _run(*arg, **kw):
				try:
					v_run = run(*arg, **kw)
				except Exception as e:
					print('trial failed: ', e)
					v_run = {}
				return v_run

			objective = partial(objective, c=c, run=_run)

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

			from things.for_torch.torch_utils import get_max_mem_c

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
			from things.for_torch.torch_utils import gen_profile
			c.mode = 'train'
			fn = partial(run, c=c, v_init=v_init)
			v_run = gen_profile(fn, c)
			return v_run

	run_mode = RunMode()

	res, v_run = dict(), dict(v_init_next=dict())

	run_mode_all = [c.mode,] if c.mode else c.multimode.split(':')
	print('run.py:mode = \n ***', run_mode_all, '***')
	for mode_i, mode in enumerate(run_mode_all):

		next_i = mode_i+1
		if next_i < len(run_mode_all):
			v_run.get('v_init_next', {})['next'] = run_mode_all[next_i]

		c_update_next = v_run.get('c_update_next', {})

		v_init_next = v_run.get('v_init_next', {})

		c.mode = mode

		v_run = run_mode(c, v_init=v_init_next, c_update=c_update_next)

		res[mode] = deepcopy(v_run)

	
