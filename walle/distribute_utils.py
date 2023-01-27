
from time import time

from pathlib import Path
from itertools import islice
from time import sleep
import optree
import os
import gc
from simple_slurm import Slurm
import sys
from typing import Callable, Tuple, Any, TypedDict
import wandb
import pprint
import inspect
import numpy as np
from copy import copy, deepcopy
import torch 
from functools import partial 
import pickle as pk
import yaml
import json
from functools import partial
import numpy as np

from .utils import dict_to_cmd, cmd_to_dict, dict_to_wandb, debug_dict
from .utils import mkdir, iterate_n_dir, gen_time_id, add_to_Path, dump, load
from .utils import get_cartesian_product, type_me, run_cmds, flat_any 
from .pyfig_utils import PlugIn, PyfigBase

from torch import nn

this_dir = Path(__file__).parent
hostname = os.environ['HOSTNAME']

class naive(PyfigBase.dist):
	dist_method: 	str		= 'naive'  # options: accelerate
	sync_step:      int     = 5


	def sync(ii, step: int, v_d: dict) -> dict:
		v_path = (ii._p.exchange_dir / f'{step}_{ii._p.dist.dist_id}').with_suffix('.pk')
		v_mean_path = add_to_Path(v_path, '-mean')
		
		try:
			gc.disable()

			v_ref_leaves, treespec = optree.tree_flatten(v_d)
			v_sync_save = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in v_ref_leaves]
			dump(v_path, v_sync_save)

		except Exception as e:
			print(e)
		finally:
			gc.enable()
		
		if ii._p.dist.head:

			n_ready = 0
			while n_ready < ii._p.resource.n_gpu:
				k_path_all = list(ii._p.exchange_dir.glob(f'{step}_*'))
				n_ready = len(k_path_all)

			for i, p in enumerate(k_path_all):
				leaves = [load(p),] if i==0   else [*leaves, load(p)]

			v_mean = [np.stack(l).mean(axis=0) for l in zip(*leaves)]

			try:
				gc.disable()
				for p in k_path_all:
					dump(add_to_Path(p, '-mean'), v_mean)
			except Exception as e:
				print(e)
			finally:
				sleep(0.01)
				[p.unlink() for p in k_path_all]
				gc.enable()

		while v_path.exists():
			sleep(0.02)
		sleep(0.02)

		gc.disable()
		try:
			v_sync_leaves = load(v_mean_path)  # Speed: Only load sync vars
			v_sync_leaves = [torch.tensor(data=v, device=ref.device, dtype=ref.dtype, requires_grad=False) 
				if isinstance(ref, torch.Tensor) else v 
				for v, ref in zip(v_sync_leaves, v_ref_leaves)]
			v_sync = optree.tree_unflatten(treespec=treespec, leaves=v_sync_leaves)
			
		except Exception as e:
			v_sync = v_d
			print(e)
		finally: # ALWAYS EXECUTED
			v_mean_path.unlink()
			gc.enable()
		return v_sync


import accelerate


class hf_accelerate(PyfigBase.dist):

	plugin_ignore: list = ['controller']

	controller: accelerate.Accelerator = None

	dist_method: 	str		= 'hf_accelerate'
	sync_step:		int		= 5
	_launch_cmd:	str  	= property(lambda _: f'accelerate launch {dict_to_cmd(_.dist_c.d, exclude_false=True)} ')

	class dist_c(PlugIn):
		# compute_environment = 'LOCAL_MACHINE'
		# distributed_type =  'MULTI_GPU'
		multi_gpu = True
		machine_rank = '0'
		same_network = True
		main_process_port = str(np.random.randint(30000, 60000))
		num_processes =  property(lambda _: str(_._p._p.resource.n_gpu))
		num_machines =  property(lambda _: str(_._p._p.resource.n_node))

	def __init__(ii, parent=None):
		super().__init__(parent)
		ii.controller: accelerate.Accelerator = accelerate.Accelerator()

	@torch.no_grad()
	def sync(ii, step, v_d: dict[str:torch.Tensor]) -> list[torch.Tensor]:

		if ((step/ii.sync_step)==1) and ii._p.debug:
			[print(k, v.shape) for k,v in v_d.items()]

		v_flat, treespec = optree.tree_flatten(v_d)
		v_sync_flat: list[torch.Tensor] = ii.dist.gather(v_flat)
		for i, (v, v_ref) in enumerate(zip(v_sync_flat, v_flat)):
			if ((step/ii.sync_step)==1) and ii._p.debug:
				print(v.shape, v_ref.shape)
			v = v.reshape(-1, *v_ref.shape).mean(dim=0)
			v_sync_mean = [v] if i==0 else [*v_sync_mean, v]
		v_sync = optree.tree_unflatten(treespec=treespec, leaves=v_sync_mean)

		return v_sync

	def backward(ii, loss: torch.Tensor):
		opt_is_adahess = ii._p.opt.opt_name.lower()=='AdaHessian'.lower()
		ii.controller.backward(loss, create_graph=opt_is_adahess)

	def dist_set_device(ii, device=None):
		print('getting devices with accelerate ', ii.controller._get_devices())
		ii._device = ii.controller.device
		return ii._device

	def dist_set_seed(ii, seed=None):
		from accelerate.utils import set_seed
		print('setting seed w accelerate ' )
		ii._seed = seed or ii._p.seed
		set_seed(ii._seed)
  
	def prepare(ii, model=None, opt=None, **kw):
		return ii.controller.prepare(model=model, opt=opt, **kw)  # docs:accelerate