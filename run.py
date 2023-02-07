import numpy as np
import wandb
import os

from typing import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from hwat import PyfigDataset

from functorch import make_functional_with_buffers, vmap
from things.for_torch.torch_utils import get_opt, get_scheduler
from things.utils import npify_tree, compute_metrix

from functools import partial

from hwat import Ansatz_fb as Model
from hwat import compute_ke_b, compute_pe_b
import numpy as np
from copy import deepcopy
import pprint

from things.utils import Metrix
from things.core_utils import lo_ve
from pyfig import Pyfig

# m_orb = model.full_det_from_spin_det(*m_orb_ud)
# orb = model.full_det_from_spin_det(*orb_ud)
# eye = torch.eye(m_orb.shape[-1], device=m_orb.device, dtype=m_orb.dtype)
# loss = ((orb**2 - m_orb**2) * eye).mean(0).sum()

# if the other loss variable is detached, then the gradients are not computed
# sometimes, the one of the variables doesn't have gradients and you forget
# Parameters which did not receive grad for rank 0: w_final.weight
# error when a weight is not included in all gather

def custom_collate(batch):
	return batch[0] 

def init_exp(c: Pyfig, state: dict= None):

	state = state or {}

	torch.cuda.empty_cache()
	torch.backends.cudnn.benchmark = c.cudnn_benchmark

	if not c.dist.ready:
		c.dist.init()

	pprint.pprint(c.d_flat)
	
	c.seed = c.dist.set_seed()
	c.dtype = c.dist.set_dtype()
	c.device = c.dist.set_device()
	c.to(framework='torch')
	c.app.init_app()

	model: torch.nn.Module = c.partial(Model).to(dtype=c.dtype)
	if state.get('model'):
		model.load_state_dict(state['model'])
		
	model_to_fn: torch.nn.Module = c.partial(Model, mol=None).to(dtype=c.dtype)
	model_fn, param, buffer = make_functional_with_buffers(model_to_fn)
	model_fn_vmap = vmap(model_fn, in_dims=(None, None, 0))
	
	dataset = PyfigDataset(c, state=state)
	dataloader = DataLoader(dataset, batch_size= c.data.loader_n_b, collate_fn= custom_collate)  # c.data.n_b otherwise because of the internal sampler

	opt: Optimizer = get_opt(**c.opt.d_flat)(model.parameters())
	if state.get('opt'):
		opt.load_state_dict(state['opt'])

	### under construction ###
	# pre_opt = get_opt(**c.opt.pre_opt.d_flat)(model.named_parameters())
	### under construction ###

	scheduler = get_scheduler(**c.scheduler.d_flat, n_scheduler_step=c.n_step)(opt)
	if state.get('scheduler'):
		scheduler.load_state_dict(state['scheduler'])

	model, dataloader, opt, scheduler = c.dist.prepare(model, dataloader, opt, scheduler)

	model.train()
	if c.eval_tag in c.mode:
		model.eval()

	compute_loss = partial(loss_fn, mode=c.mode, model_fn=model_fn_vmap, model=model)

	device = next(model.parameters()).device
	dtype = next(model.parameters()).dtype
	dataloader.dataset.init_dataset(c, device, dtype, model=model)

	return model, dataloader, compute_loss, opt, scheduler


def loss_fn(
	data: torch.Tensor=None,
	model:torch.nn.Module=None,
	model_fn: Callable=None,
	debug: bool = False,
	**kw, 
):

	v_d = dict(data= data.detach()) | {k:v.detach() for k,v in kw.items() if isinstance(v, torch.Tensor)}

	if c.app.compute_energy or c.app.loss=='vmc':
		ke = compute_ke_b(model, model_fn, data, ke_method=c.model.ke_method)
		with torch.no_grad():
			pe = compute_pe_b(data, c.app.a, c.app.a_z)
			e = pe + ke
			e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
			e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)
			energy = (e_clip - e_clip.mean())
		
		v_d |= dict(e= e, pe= pe, ke= ke)
		

	if c.app.loss=='orb_mse':

		model = c.dist.unwrap(model)
		m_orb_ud = model.compute_hf_orb(data.detach())
		orb_ud = model.compute_orb(data.detach())

		loss = sum([(torch.diagonal(o - mo, dim1=-1, dim2=-2))**2 for o, mo in zip(orb_ud, m_orb_ud)]).mean() # increasing dets can artificially boost the lr confusing understanding 

	elif c.app.loss=='vmc':

		loss = ((energy / c.app.a_z.sum()) * model(data)).mean()
		
	else: 
		loss = None

	if loss is not None:

		v_d |= dict(loss= loss.item())


		create_graph = c.opt.opt_name.lower() == 'AdaHessian'.lower()
		# print(create_graph)
		c.dist.backward(loss, create_graph=create_graph)
		v_d.setdefault('grads', {k: (p.grad.detach()) for k,p in model.named_parameters()})

	return loss, v_d 

from torch import Tensor
from torch import nn

@torch.no_grad()
def update_grads(step: int, model: nn.Module, v_d: dict, this_is_noop: bool= False, debug=False, **kw):
	
	grads = v_d.get('grads')
	if not grads:
		grads = {}

	for i, (k, p) in enumerate(model.named_parameters()):

		g = grads.get(k)
		if g is None: 
			g = torch.zeros_like(p)
		
		if p.grad is None:
			p.grad = torch.zeros_like(p)
		else:
			p.grad.copy_(g)	

		if debug and step==1:
			print('update: i,k,g,p.req_grad,p: ', i, k, g.mean(), p.requires_grad, p.mean(), sep='\n')

	return v_d

@torch.no_grad()
def update_params(model, opt, scheduler, clip_grad_norm: float= 1., **kw):
	torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
	scheduler.step()
	opt.step()
	

def smrz(**kw):
	summary = {}
	for k, v_obj in kw.items():
		if k in 'model':
			summary['n_param'] = sum(p.numel() for p in v_obj.parameters())
	return summary


def this_is_noop(step, n_step, n= None, every= None):
	if os.environ.get('debug', False) and step<10:
		print('this_is_noop: step, n_step, n, every: ', step, n_step, n, every)
		if n:
			n = min(n, n_step)
			print('n_min: ', n)
		every = every or (n_step/(n or n_step))
		print('condition: \t' , (step % every)==0)

	if every:
		if every < 0:
			return True
		elif step == n_step:
			return False
		else:
			return not ((step % every)==0)
	
	elif n:
		if n_step < n:
			return False
		elif n < 0:
			return True
		elif step == n_step:
			return False
		else:
			return not ((step % (c.n_step//n))==0)
	else:
		raise ValueError('should_log: n or every must be set')


def run(c: Pyfig=None, c_update: dict= None, **kw):

	c.update((c_update or {}) | (kw or {}))

	state: dict = lo_ve(path= c.lo_ve_path) if c.lo_ve_path else None
	state = {k:torch.tensor(v) if isinstance(v, np.ndarray) else v for k,v in (state or {}).items()}

	model, dataloader, compute_loss, opt, scheduler = init_exp(c, state)
	model.requires_grad_(True)

	init_summary = smrz(model=model)

	metrix = Metrix(c.mode, init_summary, c.opt_obj_key, opt_obj_op= c.opt_obj_op)

	lo_ve_path_fn = lambda mode, group_i, step: c.paths.state_dir / f'{mode}_{group_i}_{step}.state'
	
	c.start()
	
	print('run:go: ', c.mode, 'c_update: ', c_update, sep='\n')
	def run_loop():
		""" !!test wrap in function to force sync of dist, not clear if needed"""
		
		v_cpu_d = dict()
		for step, loader_d in enumerate(dataloader, start=1):

			model.zero_grad(set_to_none=True)

			loss, v_d = compute_loss(**loader_d, model=model, debug=c.debug)

			v_d = c.dist.sync(v_d, sync_method= c.mean_tag, this_is_noop= this_is_noop(step, c.n_step, every= c.dist.sync_step))

			update_grads(step, model, v_d, this_is_noop= not ('grads' in v_d))
			update_params(model, opt, scheduler, clip_grad_norm= 1.) # !! pyfigize
			v_d.setdefault('params', {k: (p.detach()) for k,p in model.named_parameters()})

			if c.is_logging_process:

				v_cpu_d: dict = npify_tree(v_d)

				if not this_is_noop(step, c.n_step, n= c.n_log_state):
					if not c.log_state_keys:
						c.log_state_keys = []
					elif 'all' in c.log_state_keys:
						c.log_state_keys = list(v_cpu_d.keys())
					state = dict(filter(lambda kv: kv[0] in c.log_state_keys, v_cpu_d.items()))
					lo_ve(path= lo_ve_path_fn(c.mode, c.group_i, step), data= state)
				
				if not this_is_noop(step, c.n_step, n= c.n_log_metric):
					v_metrix = metrix.tick(step, v_cpu_d= v_cpu_d)
					v_metrix = compute_metrix(v_metrix, sep= '/', debug= c.debug)
					if not 'all' in c.log_metric_keys:
						v_metrix = dict(filter(lambda kv: kv[0] in c.log_metric_keys, v_metrix.items()))
					wandb.log(v_metrix, step= step, commit= True)
	

		v_cpu_d = v_cpu_d or npify_tree(v_d)
		v_cpu_d = metrix.tock(c.n_step, v_cpu_d)
		return v_cpu_d

	v_cpu_d = run_loop()
	v_cpu_d = c.dist.sync(v_cpu_d, sync_method= c.gather_tag, this_is_noop= c.opt_hypam_tag in c.mode)
	
	c.to(framework='numpy')

	if c.dist.head:
		c.app.record_summary(summary= metrix.summary, opt_obj_all= metrix.opt_obj_all)

	c_update	= dict(lo_ve_path= lo_ve_path_fn(c.mode, c.group_i, c.n_step))
	v_run 		= dict(c_update= c_update, v_cpu_d= v_cpu_d)

	c.end()
	torch.cuda.empty_cache()

	return v_run or {}


if __name__ == "__main__":
	import traceback 

	# run output must contain c_update 
	# c_update must contain lo_ve_path
	
	allowed_mode_all = 'train:eval:max_mem:opt_hypam:profile:train-eval'

	c = Pyfig(notebook=False, sweep=None, c_update=None)

	class RunMode:

		def __call__(ii, c: Pyfig, v_run_prev: dict=None):
			
			fn = getattr(ii, c.mode)

			print('run:', c.mode, 'c_update', pprint.pformat(v_run_prev))

			# guarantee a c_update input
			v_run = fn(c=c, c_update= v_run_prev.get('c_update', {}))
			return v_run

		def opt_hypam(ii, c: Pyfig, c_update: dict):

			assert c.is_logging_process is True

			c.update(c_update or {})

			def run_trial(c: Pyfig= None, c_update_trial: dict= None):
				c_mem = c._memory()
				v_run = run(c= c, c_update= c_update_trial)
				c.update(c_mem)
				return v_run 

			v_run = c.sweep.opt_hypam(run_trial)

			v_run.get('c_update').pop(c.lo_ve_path_tag, None)  # make sure no load passed on

			print('passing on c_update from opt_hypam:')
			pprint.pprint(v_run.get('c_update', {}))
			return v_run or {}

		def max_mem(ii, c: Pyfig, **kw):
			""" potentially depreciated """
			print('\nrun:max_mem')
			from things.for_torch.torch_utils import get_max_mem_c
			
			path = c.paths.exchange_dir / 'max_mem.flag'

			if c.dist.rank==0:
				c_mem = deepcopy(c._memory())

				min_power = 5 # 2**5 = 32 walkers
				max_power = c.app.debug_c.max_power if c.debug else 20 # 2**20 = 1 million walkers
				v_run_mem = get_max_mem_c(run, c=c, min_power= min_power, max_power= max_power)

				c.update(c_mem | v_run_mem.get('c_update', {}))
				path.write_text(f'{c.data.n_b}')
			else:
				print('max_mem:waiting rank,', c.dist.rank)
				from time import sleep
				while not path.exists():
					sleep(10)
				sleep(c.dist.rank)
				c.data.n_b = int(path.read_text())
				print('max_mem:updated n_b', c.data.n_b)
			
			pprint.pprint(v_run)
			return v_run or {}

		def train(ii, c: Pyfig, c_update: dict= None):
			print('\nrun:train:')
			v_run = run(c=c, c_update= c_update)
			return v_run

		def pre(ii, c: Pyfig, c_update: dict= None):
			print('\nrun:pre:')
			v_run = run(c=c, c_update= c_update)
			return v_run
			
		def eval(ii, c: Pyfig, c_update: dict= None):
			print('\nrun:eval:')
			v_run = run(c=c, c_update= c_update)
			return v_run


	run_mode = RunMode()

	res = dict()

	from things.core_utils import flat_any
	# v_run: mode, c_update
	v_run = dict(mode= None, c_update=dict(lo_ve_path= None))
 
	
	c.multimode = c.multimode or c.mode or c.train_tag
	print('run.py:run_loop:mode: = \n ***', c.multimode, '***')
	for mode_i, mode in enumerate(c.multimode.split(':')):
		print('run.py:run_loop:mode: = \n ***', mode_i, mode, '***')

		c.mode = mode.split('-')[0]  	# todo: add support for mode-1:mode-cmd:mode-cfg-other:etc
		c.c_init['mode'] = c.mode 		# !!! this is important and buggy, need mode control flow

		base_c = flat_any(c.app.mode_c[mode])
		debug_c = flat_any(c.app.debug_c) if c.run_debug_c else dict()

		c.if_debug_print_d(base_c, msg='base_c')
		c.if_debug_print_d(debug_c, msg='debug_c')
		c.if_debug_print_d(c.c_init, msg='c_init')

		v_run['c_update'] |= (base_c | debug_c | flat_any(c.c_init))

		c.if_debug_print_d(v_run)

		v_run = run_mode(c, v_run_prev= v_run)

		if not v_run is None:
			v_run['mode'] = mode
			res[mode] = deepcopy(v_run)
		else:
			v_run = {}
