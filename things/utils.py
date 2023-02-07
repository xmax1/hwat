
import subprocess
from ast import literal_eval
from copy import deepcopy
from itertools import product
from pathlib import Path
from time import time
from typing import Any, Union, Callable
import pprint
import numpy as np
import optree
import paramiko
import torch

from time import sleep, time
from .core_utils import flat_any, ins_to_dict


class PlugIn:
	p = None
	
	ignore_base = ['ignore', 'p', 'd', 'd_flat', 'ignore_base'] # applies to every plugin
	ignore: list = [] # applies to this plugin

	def __init__(ii, parent=None):
		ii.p = parent

		ii.init_sub_cls()
		
		for k, v in ins_to_dict(ii, attr=True, ignore=ii.ignore+ii.ignore_base).items():
			setattr(ii, k, v)
  
	def init_sub_cls(ii,) -> dict:
		sub_cls = ins_to_dict(ii, sub_cls=True)
		for sub_k, sub_v in sub_cls.items():
			print('sub_ins:init:', sub_k, sub_v)
			sub_ins = sub_v(parent=ii)
			setattr(ii, sub_k, sub_ins)

	@property
	def d(ii):
		d = ins_to_dict(ii, sub_ins=True, prop=True, attr=True, ignore=ii.ignore_base + ii.ignore)
		if getattr(ii, '_prefix', None):
			_ = {k.lstrip(ii.prefix):v for k,v in d.items()}
		return d

	@property
	def d_flat(ii):
		return flat_any(ii.d)





### load (lo) and save (ve) lo_ve things 


### metrics

def collect_stats(k, v, new_d, p='tr', suf='', sep='/', sep_long='-'):
	depth = p.count('/')
	if depth > 1:
		sep = sep_long
	if isinstance(v, dict):
		for k_sub,v_sub in v.items():
			collect_stats(k, v_sub, new_d, p=(p+sep+k_sub))
	elif isinstance(v, list):
		for i, v_sub in enumerate(v):
			collect_stats(k, v_sub, new_d, p=(p+sep+k+str(i)))
	else:
		new_d[p+sep+k+suf] = v
	return new_d


### np things

def npify(v):
	return torch.tensor(v.numpy())

def numpify_tree(v: dict|list, return_flat_with_spec=False):
	if not isinstance(v, dict|list):
		if isinstance(v, torch.Tensor):
			v: torch.Tensor = v.detach().cpu().numpy()
		return v
	leaves, treespec = optree.tree_flatten(v)
	leaves = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in leaves]
	if return_flat_with_spec:
		return leaves, treespec
	return optree.tree_unflatten(treespec=treespec, leaves=leaves)

### torch things


def flat_wrap(wrap_fn: Callable) -> Callable:

	def _flat_wrap(d: dict) -> dict:
		d_v, treespec = optree.tree_flatten(d)
		d_flat = wrap_fn(d_v)
		return optree.tree_unflatten(treespec=treespec, leaves=d_flat)

	return _flat_wrap


def npify_list(d_v: list) -> dict:
	return [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in d_v]


npify_tree: Callable = flat_wrap(npify_list)

if torch:

	fancy = dict()

	def compute_metrix(v: dict|list|torch.Tensor|np.ndarray, parent='', sep='/', debug=False):

		items = {}

		if isinstance(v, list):
			all_scalar = all([np.isscalar(_vi) for _vi in v])
			if all_scalar:
				v = np.array(v)
			else:
				v = {str(i): v_item for i, v_item in enumerate(v)}
		
		if isinstance(v, dict):
			for k_item, v_item in v.items():
				k = ((parent + sep) if parent else '') + k_item
				items |= compute_metrix(v_item, parent=k, sep=sep)

		elif isinstance(v, torch.Tensor):
			v = v.detach().cpu().numpy()

		if np.isscalar(v):
			items[parent] = v

		elif isinstance(v, (np.ndarray, np.generic)):
			
			items[parent + r'_\mu$'] = v.mean()
			if v.std() and debug:
				items['std'+sep+parent + r'_\sigma$'] = v.std()
			
		return items
		


	def torchify_tree(v: np.ndarray, v_ref: torch.Tensor):
		leaves, tree_spec = optree.tree_flatten(v)
		leaves_ref, _ = optree.tree_flatten(v_ref)
		leaves = [torch.tensor(data=v, device=ref.device, dtype=ref.dtype) 
				if isinstance(ref, torch.Tensor) else v 
				for v, ref in zip(leaves, leaves_ref)]
		return optree.tree_unflatten(treespec=tree_spec, leaves=leaves)



import numpy as np
from typing import Callable

class Metrix:
	t0: 		   float = time()
	step: 		   int 	 = 0
	mode: str = None

	max_mem_alloc: float = None
	t_per_it: 	   float = None
	
	opt_obj: 	   float = None
	opt_obj_all:    list = None
	overview: 		dict = None
	exp_stats: 		dict = None

	summary: 	   dict = None
	
	def __init__(ii, 
		mode: str, 
		init_summary: dict,
		opt_obj_key: str, 
		opt_obj_op: Callable 	= None, 
		log_exp_stats_keys: list= None,
		log_eval_keys: list		= None,
	):
		
		apply_mean = lambda x: x.mean()

		ii.mode = mode

		ii.summary = init_summary 		or {}
		ii.opt_obj_key = opt_obj_key 	or 'loss'
		ii.opt_obj_op = opt_obj_op 		or apply_mean

		ii.log_exp_stats_keys = log_exp_stats_keys 	# if None, log all
		ii.log_eval_keys = log_eval_keys 			# if None, log all

		ii.opt_obj_all = []

		torch.cuda.reset_peak_memory_stats()

	def tick(ii, 
		step: int, 
		v_cpu_d: dict, 
		this_is_noop: bool = False,
	) -> dict:
		""" can only tick once per step """
		if this_is_noop:
			return {}

		dstep = step - ii.step
		ii.step = step

		ii.t_per_it, ii.t0 = (time() - ii.t0)/dstep, time()
		ii.max_mem_alloc = torch.cuda.max_memory_allocated() // 1024 // 1024
		torch.cuda.reset_peak_memory_stats()

		ii.opt_obj = ii.opt_obj_op(v_cpu_d.get(ii.opt_obj_key, np.array([0.0, 0.0])))
		ii.opt_obj_all += [ii.opt_obj,]

		ii.overview = dict(
			opt_obj= ii.opt_obj, 
			t_per_it= ii.t_per_it,
			max_mem_alloc= ii.max_mem_alloc,
			opt_obj_all= ii.opt_obj_all,
		)

		if ii.log_exp_stats_keys is None:
			log_exp_stats_keys = v_cpu_d.keys()

		log_kv = dict(filter(lambda kv: kv[0] in log_exp_stats_keys, deepcopy(v_cpu_d).items()))

		ii.exp_stats = {'exp_stats': dict(all=(log_kv or {}), overview= ii.overview)}

		return ii.exp_stats

	def tock(ii, step, v_cpu_d):
		import pprint
		if step==ii.step:
			ii.step -= 1
		_ = ii.tick(step, v_cpu_d, this_is_noop=False)
		print('', 'overview (t_per_it possibly wrong print, dist issue):', sep='\n')
		pprint.pprint({k:np.asarray(v).mean() for k,v in ii.overview.items()})
		return ii.overview

