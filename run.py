import numpy as np
import wandb
import os
from pathlib import Path
from things.core import mkdir, this_is_noop

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
from things.typfig import convert_to
from things.core import lo_ve
from things.aesthetic import print_table

from pyfig import Pyfig

from torch import Tensor
from torch import nn
import time

def custom_collate(batch):
	return batch[0] 

def init_exp(c: Pyfig, state: dict = None):

	with torch.no_grad():
		torch.cuda.empty_cache()

	torch.backends.cudnn.benchmark = c.cudnn_benchmark

	if not c.dist.ready:
		c.dist.init()

	c.seed = c.dist.set_seed()
	c.dtype = c.dist.set_dtype()
	c.device = c.dist.set_device()
	c.app.init_app()
	c.to(framework='torch')
	print_table(data= dict(seed= c.seed, dtype= c.dtype, device= c.device, rank= c.dist.rank))

	model: torch.nn.Module = c.partial(Model, device= 'cpu')
	
	if c.lo_ve_path:

		d_path = Path(c.lo_ve_path, 'd.state')
		model_path = Path(c.lo_ve_path, 'model.state')

		if d_path.exists():
			print('\n\nloading state')
			state = lo_ve(path= d_path)
			state = convert_to(state, to= 'torch', device= c.device, dtype= c.dtype)
		
		if model_path.exists():
			print('\n\nloading Model', model_path)
			model_state_dict = torch.load(model_path)
			model.load_state_dict(model_state_dict)
			model.eval()

	state = state or {}

	model = model.to(dtype= c.dtype)

	model_to_fn: torch.nn.Module = c.partial(Model, mol= None).to(dtype= c.dtype)
	model_fn, param, buffer = make_functional_with_buffers(model_to_fn)
	model_fn_vmap = vmap(model_fn, in_dims=(None, None, 0))
	
	dataset = PyfigDataset(c, state= state)
	dataloader = DataLoader(dataset, batch_size= c.data.loader_n_b, collate_fn= custom_collate)  # c.data.n_b otherwise because of the internal sampler

	opt: Optimizer = get_opt(**c.opt.d_flat)(model.parameters())
	if state.get('opt'):
		opt.load_state_dict(state.get('opt'))

	### under construction ###
	# pre_opt = get_opt(**c.opt.pre_opt.d_flat)(model.named_parameters())
	### under construction ###

	scheduler = get_scheduler(**c.scheduler.d_flat, n_scheduler_step= c.n_step)(opt)
	if state.get('scheduler'):
		scheduler.load_state_dict(state['scheduler'])

	model, dataloader, opt, scheduler = c.dist.prepare(model, dataloader, opt, scheduler)

	model.train()
	if c.eval_tag in c.mode:
		model.eval()

	compute_loss = partial(loss_fn, mode= c.mode, model_fn= model_fn_vmap, model= model)

	dataloader.dataset.init_dataset(c, device= c.device, dtype= c.dtype, model= model)

	return model, dataloader, compute_loss, opt, scheduler


def loss_fn(
	step: int,
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
			e_clip = torch.clip(e, min= e-5*e_mean_dist, max= e+5*e_mean_dist)
			energy = (e_clip - e_clip.mean())
		
		v_d |= dict(e= e, pe= pe, ke= ke)
		
	if c.app.loss=='orb_mse':

		model = c.dist.unwrap(model)

		m_orb_ud = model.compute_hf_orb(data.detach())
		m_orb_ud = [torch.tensor(mo, dtype= data.dtype, device= data.device, requires_grad= False) for mo in m_orb_ud]
		orb_ud = model.compute_orb(data.detach())

		loss = sum([(torch.diagonal(o - mo, dim1= -1, dim2= -2)**2).mean() for o, mo in zip(orb_ud, m_orb_ud)]) 
		loss *= float(step / c.n_step)

	elif c.app.loss=='vmc':

		loss = ((energy / c.app.a_z.sum()) * model(data)).mean()
		
	else: 
		loss = None

	if loss is not None:

		if step < 200:
			p = torch.exp(model(data))**2
			std = (2. * p.std()).clamp_max(1.)
			samples = torch.randn_like(p) * std + p.mean()
			kl_div_loss = torch.nn.functional.kl_div(input= p, target= samples, reduction= 'batchmean')
			
			loss += (kl_div_loss * (0.5 * loss.detach()/kl_div_loss.detach()))
			
			v_d |= dict(kl_div_loss= kl_div_loss.item(), kl_std= std.item(), kl_samples= samples.detach(), kl_p= p.detach())

		create_graph = c.opt.opt_name.lower() == 'AdaHessian'.lower()
		c.dist.backward(loss, create_graph=create_graph)

		grads = {k: p.grad.detach() for k,p in model.named_parameters()}
		v_d |= dict(loss= loss.item(), grads= grads)

	return loss, v_d 


def gaussian_clip(g: Tensor, cutoff: float= 2.):
	mean = g.mean()
	std = g.std()
	g = g - mean			# center
	sign = g.sign()			
	g_abs = g.abs()
	g_max = cutoff * std
	no_more_clip_here = g_max / 2.
	g_diff = g_abs - g_max
	g_diff.clamp_min_(0.)
	cut = torch.randn(g.shape, device= g.device, dtype= g.dtype).abs() * (g_diff / 2.)**2
	g_cut = g_max - cut
	g_cut.clamp_min_(no_more_clip_here)
	return dict(g= (g_cut * sign) + mean, g_og= g)


@torch.no_grad()
def update_grads(step: int, model: nn.Module, v_d: dict, debug= False, **kw):
	v_d['grads_clip'] = dict()

	grads = v_d.get('grads', dict())

	for i, (k, p) in enumerate(model.named_parameters()):

		g = grads.get(k)
		if g is None: 
			g = torch.zeros_like(p)
		
		g_d = gaussian_clip(g, cutoff= 1.+ 10.*float(step/c.n_step))
		g = g_d['g']

		if p.grad is None:
			p.grad = torch.zeros_like(p)
		else:
			p.grad.copy_(g)

		v_d['grads_clip'][k] = g

	return v_d

@torch.no_grad()
def update_params(model: nn.Module, opt, scheduler, clip_grad_norm: float= 1., **kw):
	# torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
	scheduler.step()
	opt.step()
	
def smrz(**kw):
	summary = {}
	for k, v_obj in kw.items():
		if k in 'model':
			summary['n_param'] = sum(p.numel() for p in v_obj.parameters())
	return summary

def run(c: Pyfig= None, c_update: dict= None, **kw):

	c.update((c_update or {}) | (kw or {}))

	model, dataloader, compute_loss, opt, scheduler = init_exp(c)
	model.requires_grad_(True)

	init_summary = smrz(model= model)

	metrix = Metrix(c.mode, init_summary, c.opt_obj_key, opt_obj_op= c.opt_obj_op)

	lo_ve_dir_fn = lambda mode, group_i, step: mkdir(c.paths.state_dir / f'{mode}_{group_i}_i{step}')

	c.start()
	
	print('run:go: ', c.mode, 'c_update: ', c_update, sep= '\n')
	def run_loop(model: nn.Module):
		""" !!test wrap in function to force sync of dist, not clear if needed"""
		
		v_cpu_d = dict()
		for step, loader_d in enumerate(dataloader, start= 1):

			model.zero_grad(set_to_none= True)

			loss, v_d = compute_loss(step, **loader_d, model= model, debug= c.debug)

			v_d = c.dist.sync(
				v_d, 
				sync_method= c.mean_tag, 
				this_is_noop= this_is_noop(step, c.n_step, every= c.dist.sync_every)
			)

			v_d = update_grads(step, model, v_d, debug= c.debug)
			update_params(model, opt, scheduler)
			
			v_d.update( (('params', {k:p.detach() for k,p in model.named_parameters()}),) )

			if c.is_logging_process:

				v_cpu_d: dict = npify_tree(v_d)

				if not this_is_noop(step, c.n_step, n= c.n_log_state):
					lo_ve(path= lo_ve_dir_fn(c.mode, c.group_i, step) / 'd.state', data= v_cpu_d)
					torch.save(model.cpu().state_dict(), lo_ve_dir_fn(c.mode, c.group_i, step) / 'model.state')
				
				if not this_is_noop(step, c.n_step, n= c.n_log_metric):
					v_metrix = metrix.tick(step, v_cpu_d= v_cpu_d)
					v_metrix = compute_metrix(v_metrix, sep= '/', debug= c.debug)
					wandb.log(v_metrix, step= step, commit= True)  # possibly change because of the gpu logging

		print_table(dict(mode= c.mode, step= step, n_step= c.n_step))
		v_cpu_d = npify_tree(v_d)
		v_cpu_d = metrix.tock(c.n_step, v_cpu_d)
		return v_cpu_d

	v_cpu_d = run_loop(model)
	
	c.to(framework= 'numpy')

	if c.dist.head:
		c.app.record_summary(summary= metrix.summary, opt_obj_all= metrix.opt_obj_all)

	print('lo_ve_path:', lo_ve_dir_fn(c.mode, c.group_i, c.n_step))
	c_update	= dict(lo_ve_path= lo_ve_dir_fn(c.mode, c.group_i, c.n_step))
	v_run 		= dict(c_update= c_update, v_cpu_d= v_cpu_d)

	c.end()
	
	return v_run or {}


if __name__ == "__main__":

	c = Pyfig(notebook=False, sweep=None, c_update=None)

	v_run = dict(mode= None, c_update=dict(lo_ve_path= None))
 
	
	c.multimode = c.multimode or c.mode or c.train_tag
	print('run.py:run_loop:mode: = \n ***', c.multimode, '***')

	mem = c._memory()

	for mode_i, mode in enumerate(c.multimode.split(':')):
		print('run.py:run_loop:mode: = \n ***', mode_i, mode, '***')

		c_update = v_run.get('c_update', {})
		c.update_mode(mode= mode, c_update= c_update)

		if mode in 'train:pre:eval':
			if mode == 'pre':
				c.lo_ve_path = None

			v_run = run(c= c)

		elif mode == 'opt_hypam':

			print('pyfig: ', c)

			def run_trial(c: Pyfig= None, c_update_trial: dict= None):
				# what is the c object here? A new reference, or something else? 
				c_mem = c._memory()

				c.update_mode('pre', c_update= c_update_trial, sync_every= 0, n_log_metric = 100, n_log_state= 1, n_pre_step= c.n_opt_hypam_step)
				v_run = run(c= c)

				print('\n\n\n V_RUN:')
				pprint.pprint(v_run, indent= 4, width= 100)

				lo_ve_path = v_run.get('c_update', {}).get('lo_ve_path')
				c.update_mode('opt_hypam', lo_ve_path= lo_ve_path, sync_every= 0, n_log_metric= 20)
				v_run = run(c= c)

				c.update(c_mem)
				return v_run 

			v_run = c.sweep.opt_hypam(run_trial)

			v_run.get('c_update').pop(c.lo_ve_path_tag, None)  # make sure no load passed on

			print('passing on c_update from opt_hypam:')
			pprint.pprint(v_run.get('c_update', {}))
		
		c.update(mem)
			


			

"""

def max_mem(ii, c: Pyfig, **kw):
 potentially depreciated 
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

"""