import sys
from typing import Callable
from pathlib import Path
import wandb
from functools import partial 

import numpy as np

import torch
import optree

from ..core_utils import flat_any
from ..pyfig_utils import PyfigBase 

from torch import nn


def try_convert(k, v, device, dtype):
	try:
		v = v.to(dtype)
		print(f'{k} to dtype', dtype)
	except Exception as e:
		print(f'\n*not* converting {k} to dtype', dtype)
	try:
		v = v.to(device)
		print(f'\n{k} to device', device)
	except Exception as e:
		print(f'\n*not* {k} to device', device)
	return v
	

def gen_profile(
	fn: Callable,
	c: PyfigBase, 
	wait=1, 
	warmup=1, 
	active=1, 
	repeat=1,
	**kw
) -> dict:
	print('profile: ', fn)

	profiler = torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
		schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
		on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
		profile_memory=True, with_stack=True, with_modules=True
	)
	with profiler:
		for _ in range((wait + warmup + active) * repeat):
			fn()
			profiler.step()

	profiler.export_stacks(profile_dir/'profiler_stacks.txt', 'self_cuda_time_total')

	print(profiler.key_averages().table())

	profile_art = wandb.Artifact(f"trace", type="profile")
	p = next(profile_dir.iterdir())
	profile_art.add_file(p, "trace.pt.trace.json")
	profile_art.save()
	return init_d


def flat_dict(d:dict, items:list[tuple]=None):
	items = items or []
	for k,v in d.items():
		if isinstance(v, dict):
			items.extend(flat_dict(v, items=items).items())
		else:
			items.append((k, v))
	return dict(items)
	

def get_max_mem_c(fn: Callable, c: PyfigBase, min_power=8, max_power=20, **kw) -> dict:
	print('\nget_max_mem_c:total')
	
	import traceback

	t = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
	r = torch.cuda.memory_reserved(0)
	a = torch.cuda.memory_allocated(0)

	print('memory on device: ', t)

	for n_b_power in range(min_power, max_power):
		try:
			
			n_b = n_b=2**n_b_power
			c.update(dict(n_b= n_b))

			v_run = fn(c=c)

			v_cpu_d = flat_any(v_run[c.tag.v_cpu_d])
			keys = [k for k in v_cpu_d.keys() if c.tag.max_mem_alloc in k]
			max_mem_alloc = v_cpu_d[keys[0]]
			print(f'n_b {n_b} used {max_mem_alloc} out of {t}')
			
			if max_mem_alloc > t/2:
				break

		except Exception as e:
			print(traceback.format_exc())
			print('error: e', v_run)
			return v_run or {}
	import pprint
	print('n_b_max: ', n_b)
	v_run[c.tag.c_update].update(dict(n_b= n_b))
	print('v_run')
	v_run = v_run or {}
	c.debugger(v_run)
	return v_run

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

  
def get_opt(
	*,
	opt_name: str = None,
	lr: float = None,
	betas: tuple[float] = None,
	eps: float = None,
	weight_decay: float = None,
	hessian_power: float = None,
	default: str = 'RAdam',
	**kw, 
) -> torch.optim.Optimizer:

	if opt_name.lower() == 'RAdam'.lower():
		opt_for_model = partial(torch.optim.RAdam, lr=lr)

	elif opt_name.lower() == 'Adahessian'.lower():
		import torch_optimizer  # pip install torch_optimizer
		opt_for_model = partial(
    		torch_optimizer.Adahessian,
			lr 			 = lr,
			betas		 = betas,
			eps			 = eps,
			weight_decay = weight_decay,
			hessian_power= hessian_power
    	)
	else:
		print(f'!!! opt {opt_name} not available, returning {default}')
		opt_for_model = get_opt(opt_name=default, lr=0.001)

	return opt_for_model



def get_scheduler(
	sch_name: str = None,
	sch_max_lr: float = None,
	sch_epochs: int = None, 
	n_scheduler_step: int = None,
	default: str = 'OneCycleLR',
	**kw,
) -> torch.optim.lr_scheduler._LRScheduler:

	if sch_name.lower() == 'OneCycleLR'.lower():
		scheduler = partial(
			torch.optim.lr_scheduler.OneCycleLR, max_lr=sch_max_lr, steps_per_epoch=n_scheduler_step, epochs=sch_epochs
		)

	else:
		print(f'!!! Scheduler {sch_name} not available, returning OneCycleLR ')
		return get_scheduler(scheduler_name=default, max_lr=sch_max_lr, epochs=sch_epochs, n_step=n_scheduler_step)

	return scheduler
