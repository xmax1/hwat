import numpy as np
import wandb

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

from things.pyfig_utils import lo_ve
from things.utils import Metrix
from pyfig import Pyfig 


def init_exp(c: Pyfig, state: dict= None):

	state = state or {}

	torch.cuda.empty_cache()
	torch.backends.cudnn.benchmark = c.cudnn_benchmark

	if not c.dist.ready:
		c.dist.init()

	c.set_seed()
	c.set_dtype()
	c.set_device()
	c.to(framework='torch')
	c.app.init_app()

	model: torch.nn.Module = c.partial(Model).to(dtype=c.dtype)
	if state.get('model'):
		model.load_state_dict(state['model'])
	
	model_to_fn: torch.nn.Module = c.partial(Model, mol=None).to(dtype=c.dtype)
	model_fn, param, buffer = make_functional_with_buffers(model_to_fn)
	model_fn_vmap = vmap(model_fn, in_dims=(None, None, 0))

	### under construction ###
	dataset = PyfigDataset(c, state=state)
	def custom_collate(batch):
		return batch[0] 
	dataloader = DataLoader(dataset, batch_size= 1, collate_fn= custom_collate)  # c.data.n_b otherwise because of the internal sampler
	### under construction ###

	opt: Optimizer = get_opt(**c.opt.d_flat)(model.parameters())
	if state.get('opt'):
		opt.load_state_dict(state['opt'])

	### under construction ###
	# pre_opt = get_opt(**c.opt.pre_opt.d_flat)(model.named_parameters())
	### under construction ###
	scheduler = get_scheduler(**c.opt.scheduler.d_flat, n_scheduler_step=c.n_step)(opt)
	if state.get('scheduler'):
		scheduler.load_state_dict(state['scheduler'])

	model, dataloader, opt, scheduler = c.dist.prepare(model, dataloader, opt, scheduler)

	model.train()
	if c.tag.eval in c.mode:
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

		# if the other loss variable is detached, then the gradients are not computed
		# sometimes, the one of the variables doesn't have gradients and you forget
		# Parameters which did not receive grad for rank 0: w_final.weight
		# error when a weight is not included in all gather
	model: Model

	if c.tag.pre in c.mode:

		model = c.dist.unwrap(model)

		m_orb_ud = model.compute_hf_orb(data.detach())
		orb_ud = model.compute_orb(data.detach())
		
		# m_orb = model.full_det_from_spin_det(*m_orb_ud)
		# orb = model.full_det_from_spin_det(*orb_ud)
		# eye = torch.eye(m_orb.shape[-1], device=m_orb.device, dtype=m_orb.dtype)
		# loss = ((orb**2 - m_orb**2) * eye).mean(0).sum()
		
		loss = sum([(o - mo)**2 for o, mo in zip(orb_ud, m_orb_ud)]).mean(0).sum() / c.app.n_e

	elif c.tag.train in c.mode:
		loss = ((energy / c.app.a_z.sum()) * model(data)).mean()
	
	else:
		loss, grads, params, loss_debug = None, None, None, {}

	return loss, dict(loss= loss, e= e, pe= pe, ke= ke) 


@torch.no_grad()
def update(model, grads, opt, scheduler, step=0, debug=False):
	for i, (k, p) in enumerate(model.named_parameters()):
		g = grads.get(k)

		if g is None: # p.grad = torch.zeros_like(p) doesn't work
			p.grad = torch.zeros_like(p)
			g = torch.zeros_like(p)
		
		if debug and step==1:
			print('update: i,k,g,p.req_grad,p: ', i, k, g.mean(), p.requires_grad, p.mean(), sep='\n')
		
		p.grad.copy_(g)

	torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

	scheduler.step()
	opt.step()


from things.utils import flat_any


def smrz(**kw):
	summary = {}
	for k, v_obj in kw.items():
		if k in 'model':
			summary['n_param'] = sum(p.numel() for p in v_obj.parameters())
	return summary


def determine_log_step(step, n_step, n_log_metric):
	if step==1:
		print('determine_log_step: step, n_step, n_log_metric: ', step, n_step, n_log_metric, sep='\t')

	if n_log_metric==0:
		log = False
	elif n_log_metric<n_step:
		assert step <= n_step
		log = True
	else:
		log = (step % (c.n_step//c.n_log_metric))==0
	return log


def run(c: Pyfig=None, v_init: dict= None, c_update: dict= None, **kw):

	c.update((c_update or {}) | (kw or {}))

	state: dict = lo_ve(path= c.lo_ve_path) if c.lo_ve_path else None
	state = {k:torch.tensor(v) if isinstance(v, np.ndarray) else v for k,v in (state or {}).items()}

	model, dataloader, compute_loss, opt, scheduler = init_exp(c, state)
	model.requires_grad_(True)

	init_summary = smrz(model=model)

	metrix = Metrix(c.mode, init_summary, c.opt_obj_key, opt_obj_op= c.opt_obj_op)

	print('depreciating v_init: keys=', (v_init or {}).keys())
	print('run: ', c.mode, 'c_update: ', c_update, 'v_init:', v_init, sep='\n')

	v_cpu_d = dict()
	v_init 	= v_init or dict()

	c.start()
	lo_ve_path_fn = lambda mode, group_i, step: c.state_dir / f'{mode}_{group_i}_{step}.state'
	
	def run_loop():
		""" wrap in function to force sync """
		for step, loader_d in enumerate(dataloader, start=1):

			model.zero_grad(set_to_none=True)

			loss, v_d = compute_loss(loader_d['data'], model=model, debug=c.debug)

			if c.tag.train in c.mode:
				create_graph = c.opt.opt_name.lower()=='AdaHessian'.lower()
				c.dist.backward(loss, create_graph=create_graph)
				v_d['grads'] = {k: (p.grad.detach()) for k,p in model.named_parameters()}


			v_d |= deepcopy(loader_d)
			v_d = c.dist.sync(v_d, sync_method=c.tag.mean, this_is_noop= not (step % c.dist.sync_step))


			if c.tag.train in c.mode and c.dist.head:
				update(model, v_d['grads'], opt, scheduler, step, c.debug)
				v_d['grads'] = {k:g.detach().cpu().numpy() for k,g in v_d['grads'].items()}
				v_d['params'] = {k:p.detach().cpu().numpy() for k,p in model.named_parameters()}

			### start : logging : cpu only ###
			log_metrix = determine_log_step(step, c.n_step, c.n_log_metric)
			log_state = determine_log_step(step, c.n_step, c.n_log_state)
			is_first_or_middle_or_final = step==1 or step==c.n_step or step==c.n_step//2

			v_cpu_d: dict = npify_tree(v_d)

			if is_first_or_middle_or_final:
				print(f'{c.mode} step {step} of {c.n_step} done')
				first_and_last = {}
				for k,v in flat_any(v_cpu_d).items():
					if v is None:
						first_and_last[k] = None
					elif isinstance(v, (np.ndarray, np.generic)):
						first_and_last[k] = float(v.mean())
					else:
						first_and_last[k] = float(v)
				pprint.pprint(first_and_last)


			if c.is_logging_process:

				if log_state:

					if c.tag.eval in c.mode:
						state = dict(e= v_cpu_d.get('e'), r= v_cpu_d.get('r'))
					else:
						state = deepcopy(v_cpu_d)
					
					lo_ve(path= lo_ve_path_fn(c.mode, c.group_i, step), data= state)
				
				if log_metrix:
					v_metrix = metrix.tick(step, v_cpu_d= v_cpu_d)
					v_metrix = compute_metrix(v_metrix, sep= '/', debug= c.debug) 
					wandb.log(v_metrix, step= step, commit= True)
	
		return v_cpu_d


	v_cpu_d = run_loop()
	v_cpu_d = metrix.tock(c.n_step, v_cpu_d)

	c.to(framework='numpy')

	if c.dist.head:
		c.app.record_summary(summary= metrix.summary, opt_obj_all= metrix.opt_obj_all)

		### under construction ###
		# if c.log_data_dump_keys is not None:
 		# 	v_data_dump: dict = c.dist.sync(v_cpu_d['summary'], sync_method=c.tag.gather) 
		# 	data_dump = dict(filter(lambda kv: kv[0] in c.log_data_dump_keys, v_data_dump.items()))
		# 	lo_ve(path=(c.exp_dir/c.run_id).with_suffix('.npz'), data=data_dump)
		### under construction ###

	v_init= dict(lo_ve_path= lo_ve_path_fn(c.mode, c.group_i, c.n_step))
	v_run = dict(v_init= v_init, c_update= v_init, v_cpu_d= v_cpu_d)

	c.end()
	torch.cuda.empty_cache()

	return v_run or {}  # !!! where None?


if __name__ == "__main__":

	import traceback
	
	allowed_mode_all = 'train:eval:max_mem:opt_hypam:profile:train-eval'

	c = Pyfig(notebook=False, sweep=None, c_update=None)
	c_d0 = c.d_flat

	class RunMode:

		def __call__(ii, c: Pyfig, v_run_prev: dict=None):

			fn = getattr(ii, c.mode.split('-')[0])

			print('run:', c.mode, 'c_update', pprint.pformat(v_run_prev))

			v_run = fn(c=c, v_init= v_run_prev.get('v_init'), c_update= v_run_prev.get('c_update'))
			return v_run

		def opt_hypam(ii, c: Pyfig, v_init: dict= None, c_update: dict= None):

			from things.opt_hypam_utils import opt_hypam, objective
			import os

			
			# os.environ['RANK'] = '0'
			assert c.dist.dist_name=='naive'
			assert c.is_logging_process is True

			def run_trial(c, v_init_trial: dict= None, c_update_trial: dict= None):
				try:
					# trial overloads the 
					v_run = run(c, v_init= v_init | v_init_trial, c_update= c_update | c_update_trial)
				except Exception as e:
					print('trial failed: ', e)
					v_run = {}
				return v_run 

			from copy import deepcopy
			sync_step = deepcopy(c.dist.sync_step)
			c.dist.sync_step = -2

			objective = partial(objective, run_trial= run_trial, c=c)
			v_run = opt_hypam(objective, c)

			c.dist.sync_step = sync_step
			return v_run or {}

		def max_mem(ii, c: Pyfig, v_init: dict= None, c_update: dict= None):
			print('\nrun:max_mem')
			from things.for_torch.torch_utils import get_max_mem_c
			
			path = c.exchange_dir / 'max_mem.flag'

			if c.dist.rank==0:
				c_mem = deepcopy(c._memory())
				c.n_train_step = 30
				c.mode = c.tag.train

				min_power = 5 # 2**5 = 32 walkers
				max_power = c.debug_c.max_power if c.debug else 20 # 2**20 = 1 million walkers
				v_run = get_max_mem_c(run, c=c, min_power= min_power, max_power= max_power)

				c.update(c_mem | v_run.get('c_update', {}))
				path.write_text(f'{c.n_b}')
			else:
				print('max_mem:waiting rank,', c.dist.rank)
				from time import sleep
				while not path.exists():
					sleep(10)
				sleep(c.dist.rank)
				c.n_b = int(path.read_text())
				print('max_mem:updated n_b', c.n_b)
			
			pprint.pprint(v_run)
			return v_run or {}

		def train(ii, c: Pyfig, v_init: dict= None, c_update: dict= None):
			print('\nrun:train')
			v_run = run(c=c, v_init= v_init, c_update= c_update)
			return v_run

		def pre(ii, c: Pyfig, v_init: dict= None, c_update: dict= None):
			print('\nrun:pre')
			v_run = run(c=c, v_init= v_init, c_update= c_update)
			return v_run
			
		def eval(ii, c: Pyfig, v_init: dict= None, c_update: dict= None):
			print('\nrun:eval')
			v_run = run(c=c, v_init= v_init, c_update= c_update)
			return v_run


	run_mode = RunMode()

	res = dict()

	# v_run: mode, v_init, c_update
	v_run = dict(mode= None, v_init=dict(), c_update=dict())

	mode_all = (c.mode or c.multimode or c.tag.train).split(':') 
	
	print('run.py:run_loop:mode: = \n ***', mode_all, '***')
	for mode_i, mode in enumerate(mode_all):
		print('run.py:run_loop:mode: = \n ***', mode, '***')

		c.mode = mode

		v_run = run_mode(c, v_run_prev= v_run)

		if not v_run is None:
			v_run['mode'] = mode
			res[mode] = deepcopy(v_run)
		else:
			v_run = {}
