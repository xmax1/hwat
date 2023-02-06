
import traceback
from pathlib import Path
import os
import sys
from typing import Callable, Any
import wandb
import pprint
import inspect
import numpy as np
from copy import deepcopy
from functools import partial 
import pickle as pk
import yaml
import json
from functools import partial
import numpy as np

from .utils import dict_to_cmd, cmd_to_dict
from .utils import mkdir, iterate_n_dir, gen_time_id
from .utils import type_me, run_cmds, flat_any 

this_file_path = Path(__file__) 

from things.dist_repo import DistBase, Naive, HFAccelerate, SingleProcess
from things.logger_repo import LoggerBase, Wandb
from things.resource_repo import ResourceBase, Niflheim
from things.sweep_repo import SweepBase, Optuna
from things.gen_repo import OptBase, DataBase, SchedulerBase, PathsBase, ModelBase

class PyfigBase:

	user: 				str 	= None
 
	project:            str     = ''
	run_name:       	Path	= ''
	exp_name:       	str		= '' # default is demo
	exp_id: 			str		= ''
	run_id:		        str		= ''
	group_exp: 			bool	= False

	multimode: 			str		= '' # 'max_mem:profile:opt_hypam:train:eval'
	mode: 				str		= ''
	debug: 				bool    = False
	submit: 			bool 	= False
	
	seed:           	int   	= 0
	dtype:          	str   	= ''
	device: 		int|str 	= ''

	n_step:         	int   	= 0
	n_eval_step:        int   	= 0
	n_pre_step:    		int   	= 0

	n_log_metric:		int  	= 100
	n_log_state:		int  	= 4
	is_logging_process: bool    = False

	lo_ve_path:			str 	= ''

	_group_i: int = 0
	@property
	def group_i(ii):
		return ii._group_i

	zweep: str = ''

	n_default_step: 	int 	= 1000
	n_train_step:   	int   	= 0
	n_pre_step:    		int   	= 0
	n_eval_step:        int   	= 0
	n_opt_hypam_step:   int   	= 0
	n_max_mem_step:     int   	= 0

	@property
	def n_step(ii):
		return dict(
			train		= ii.n_train_step, 
			pre			= ii.n_pre_step, 
			eval		= ii.n_eval_step, 
			opt_hypam	= ii.n_opt_hypam_step, 
			max_mem		= ii.n_max_mem_step
		).get(ii.mode) or ii.n_default_step
  
	class sweep(SweepBase):
		pass

	class logger(LoggerBase):
		pass

	class dist(DistBase):
		pass

	class resource(ResourceBase):
		pass

	class data(DataBase):
		pass

	class opt(OptBase):
		pass

	class scheduler(SchedulerBase):
		pass

	data_tag: str		= 'data'
	
	max_mem_alloc_tag: str = 'max_mem_alloc'
	opt_obj_all_tag: str = 'opt_obj_all'
		
	pre_tag: str = 'pre'
	train_tag: str = 'train'
	eval_tag: str = 'eval'
	opt_hypam_tag: str = 'opt_hypam'
		
	v_cpu_d_tag: str = 'v_cpu_d'
	c_update_tag: str = 'c_update'
		
	lo_ve_path_tag: str = 'lo_ve_path'
	gather_tag: str = 'gather'
	mean_tag: str = 'mean'
	
	class paths(PathsBase):
		pass

	ignore_f = ['commit', 'pull', 'backward']
	ignore_p = ['parameters', 'scf', 'tag', 'mode_c']
	ignore: list = ['ignore', 'ignore_f', 'ignore_c'] + ignore_f + ignore_p
	ignore += ['d', 'cmd', 'sub_ins', 'd_flat', 'repo', 'name']

	repo = dict(
		dist = dict(
			Naive = Naive,
			HFAccelerate = HFAccelerate,
			SingleProcess = SingleProcess,
		),
		logger = dict(Wandb = Wandb),
		resource = dict(Niflheim = Niflheim),	
		sweep = dict(Optuna = Optuna),
		opt = dict(OptBase = OptBase),
		data = dict(DataBase = DataBase),
		scheduler = dict(SchedulerBase = SchedulerBase),
		paths = dict(PathBase = PathsBase),
		model = dict(ModelBase = ModelBase),
	) 

	def __init__(ii, 
		notebook:bool=False,  		# removes sys_arg for notebooks
		sweep: dict={},				# special properties for config update so is separated
		c_init: dict|str|Path={},  	# args specific
		post_init_arg: dict={},
		**other_arg
	):

		ii.init_sub_cls()

		for k,v in (sweep or {}).items():
			setattr(ii.sweep, k, v)

		if not notebook:
			ref_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, ignore=ii.ignore)
			sys_arg = cmd_to_dict(sys.argv[1:], ref_d)

		c_update = flat_any((c_init or {})) | flat_any((other_arg or {})) | (sys_arg or {})

		ii.update(c_update)
		
		if ii.debug:
			os.environ['debug'] = 'True'

		err_path = Path(ii.paths.exp_dir, str(ii.dist.rank) + '.pyferr')
		os.environ['PYFERR_PATH'] = str(err_path)

		import atexit

		def exit_handler():
			job_id = os.environ.get('SLURM_JOB_ID', False)
			if job_id:
				run_cmds(f'scancel {job_id}')
			exc_type, exc_value, exc_traceback = sys.exc_info()
			traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
			print(exc_type, exc_value, exc_traceback)
			if traceback_string:
				err_path.write_text(traceback_string)

		atexit.register(exit_handler)
		
		ii.if_debug_log([sys_arg, dict(os.environ.items()), ii.d], 
		[f'log_sys_arg_{ii.dist.pid}.log', f'log_env_run_{ii.dist.pid}.log', f'log_d_{ii.dist.pid}.log'])

	def deepcopy(ii):
		d = ii.super()(c_init=deepcopy(ii.d))
		return d

	def start(ii):

		if ii.is_logging_process:
			print('pyfig:start: logging process creating logger')
			assert ii.submit == False

			if not ii.exp_id:
				if not ii.paths.exp_dir.exists():
					print('pyfig:start: creating exp_dir, current exp_id= ', ii.exp_id)
					ii.setup_exp_dir(group_exp= ii.group_exp, force_new_id= False)

			ii._group_i += 1
			ii.run_id = ii.exp_id + '.' + ii.mode + '.' + str(ii._group_i) + '.' + str(ii.dist.rank)
		
			# ii.run_id = '.'.join(ii.run_id.split('.'))  
			
			print('pyfig:start: exp_dir = \n ***', ii.paths.exp_dir, '***')
			print('pyfig:start: run_id = \n ***', ii.run_id, '***')
			tags = [str(s) for s in [*ii.mode.split('-'), ii.exp_id, ii._group_i]]
			print(f'pyfig:logger: tags- {tags}')

			ii.logger.start(ii.d, tags=tags, run_id=ii.run_id)

	def end(ii, plugin: str= None):
		if plugin is None:
			ii.logger.end()
			ii.dist.end()
		else:
			plugin = getattr(ii, plugin).end()


	def run_submit(ii):
		try: 
			print('submitting to cluster')

			run_or_sweep_d = ii.get_run_or_sweep_d()

			ii.submit = False # docs:submit
			ii.run_sweep = False # docs:submit
			ii.zweep = False

			for i, run_d in enumerate(run_or_sweep_d):

				ii.setup_exp_dir(group_exp= False, force_new_id= True)
				ii.save_code_state()

				ii.update(run_d)
				c_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, 
					ignore=ii.ignore+['dist_c','slurm_c','parameters'])

				ii.resource.cluster_submit(c_d)
				
				ii.if_debug_log([dict(os.environ.items()), run_d], ['log-submit_env.log', 'log-submit_d.log'])

				print('log_group_url: \t\t\t',  ii.logger.log_run_url)
				print('exp_dir: \t\t\t', ii.paths.exp_dir)
				print('exp_log: \t\t\t', ii.resource.device_log_path(0))


			sys.exit('Success, exiting from submit.')
		
		except Exception as e:
			sys.exit(str(e))


	def save_code_state(ii, exts = ['.py', '.ipynb', '.md']):
		import shutil
		shutil.copytree('things', ii.paths.code_dir/'things')
		[shutil.copyfile(p, ii.paths.code_dir/p.name) for p in ii.paths.project_dir.iterdir() if p.suffix in exts]
		[shutil.copyfile(p, ii.paths.code_dir/p.name) for p in ii.paths.dump_dir.iterdir() if p.suffix in exts]

	@property
	def _paths(ii):
		path_filter = lambda item: any([p in item[0] for p in ['path', 'dir']])
		paths = dict(filter(path_filter, ii.d.items()))
		ii.if_debug_print_d(paths, '\npaths')
		return paths

	@property
	def cmd(ii):
		return dict_to_cmd(ii.d_flat)

	@property
	def d(ii):
		return ins_to_dict(ii, sub_ins=True, prop=True, attr=True, ignore=ii.ignore)

	@property
	def d_flat(ii):
		return flat_any(ii.d)

	def init_sub_cls(ii, sub_cls: type=None, name: str=None) -> dict:
		if name and not (sub_cls is None):
			print('adding sub_cls: ', name, sub_cls)
			sub_ins = sub_cls(parent=ii)
			setattr(ii, name, sub_ins)
		else:
			sub_cls = ins_to_dict(ii, sub_cls=True)
			for sub_k, sub_cls_i in sub_cls.items():
				sub_ins = sub_cls_i(parent=ii)
				setattr(ii, sub_k, sub_ins)

	def setup_exp_dir(ii, group_exp=False, force_new_id=False):

		if (not ii.exp_id) or force_new_id:
			ii.exp_id = gen_time_id(7)

			exp_name = ii.exp_name or 'junk'
			group_exp = group_exp or ('~' in exp_name)
			exp_group_dir = Path(ii.paths.dump_exp_dir, exp_name)
			exp_group_dir = iterate_n_dir(exp_group_dir, group_exp= group_exp) # pyfig:setup_exp_dir does not append -{i} if group allowed
			ii.exp_name = exp_group_dir.name

		print('exp_dir: ', ii.paths.exp_dir) 
		[mkdir(p) for _, p in ii.paths.d.items()]
	
	def get_run_or_sweep_d(ii,):
		if ii.zweep: 
			zweep = ii.zweep.split('-')
			zweep_v = zweep[0]
			t = zweep[-1]
			print('pyfig:zweep: ', zweep_v, t, zweep[1:-1])  # !!! include type me 
			return [{zweep_v: ([i] if t=='list' else int(i))} for i in zweep[1:-1]] 
		
		elif ii.run_sweep:
			return ii.sweep.get_sweep()

		else:
			""" single run takes c from base in submit loop """
			c = [dict(),] 
			return c

	def if_debug_log(ii, d_all:list, name_all: list):
		try:
			if ii.debug:
				for d, name in zip(d_all, name_all):
					if Path(ii.paths.exp_dir).is_dir():
						ii.log(d, path=ii.paths.cluster_dir/name, group_exp=False)
		except Exception as e:
			print('pyfig:if_debug_log: ', e)
			
	def partial(ii, f:Callable, args=None, **kw):
		d: dict = flat_any(args if args else {}) | ii.d_flat | (kw or {})
		d_k = inspect.signature(f.__init__).parameters.keys()
		d = {k:v for k,v in d.items() if k in d_k} 
		print('pyfig:partial setting for ', f.__name__, )
		pprint.pprint(d)
		return f(**d)

	def update(ii, _arg: dict=None, c_update: dict=None, silent: bool= False, **kw):
		silent = silent or ii.debug

		print('\npyfig:update')

		c_update = (_arg or {}) | (c_update or {}) | (kw or {})
		c_update = flat_any(c_update)
		c_keys = list(ii.d_flat.keys())
		arg = dict(filter(lambda kv: kv[0] in c_keys, c_update.items()))


		for k_update, v_update in deepcopy(arg).items():

			if k_update=='dtype': # !!! pyfig:fix: Special case, generalify
				ii.dtype = v_update
				if not silent:
					print('dtype: --- ', v_update)
			elif k_update in ii.repo.keys():
				plugin = ii.repo[k_update][v_update]
				plugin_name = k_update.split('_')[0]
				if not silent:
					print('adding plugin:', plugin_name, plugin, '...')
				ii.init_sub_cls(sub_cls=plugin, name=plugin_name)
			else:
				is_updated = walk_ins_tree(ii, k_update, v_update)
				if not is_updated:
					print(f'not updated: k={k_update} v={v_update} type={type(v_update)}')

		not_arg = dict(filter(lambda kv: kv[0] not in c_keys, arg.items()))
		ii.if_debug_print_d(not_arg, msg='\npyfig:update: arg not in pyfig')		
		
	def if_debug_print_d(ii, d: dict, msg: str=''):
		if ii.debug:
			print(msg)
			pprint.pprint(d)
		
	@staticmethod
	def log(info: dict|str, path: Path, group_exp: bool=True):
		mkdir(path)
		info = pprint.pformat(info)
		path = iterate_n_dir(path, group_exp=group_exp)
		with open(path, 'w') as f:
			f.writelines(info)

	def to(ii, framework='torch'):
		import torch
		base_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, ignore=ii.ignore+['parameters'])
		d = {k:v for k,v in base_d.items() if isinstance(v, (np.ndarray, np.generic))}
		if 'torch' in framework.lower():
			d = {k:torch.tensor(v, requires_grad=False).to(device=ii.device, dtype=ii.dtype) for k,v in d.items()}
		if 'numpy' in framework.lower():
			d = {k:v.detach().cpu().numpy() for k,v in d.items() if isinstance(v, torch.Tensor)}
		ii.update(d)

	def _memory(ii, sub_ins=True, attr=True, _prop=False, _ignore=[]):
		return ins_to_dict(ii, attr=attr, sub_ins=sub_ins, prop=_prop, ignore=ii.ignore+_ignore+['logger'])

class PlugIn:
	p: PyfigBase = PyfigBase
	
	ignore = ['ignore', 'd', 'd_flat'] # applies to every plugin

	def __init__(ii, parent=None):
		ii.p: PyfigBase = parent

		ii.init_sub_cls()
		
		for k, v in ins_to_dict(ii, attr=True, ignore=ii.ignore).items():
			setattr(ii, k, v)
  
	def init_sub_cls(ii,) -> dict:
		sub_cls = ins_to_dict(ii, sub_cls=True)
		for sub_k, sub_v in sub_cls.items():
			sub_ins = sub_v(parent=ii)
			setattr(ii, sub_k, sub_ins)

	@property
	def d(ii):
		d = ins_to_dict(ii, sub_ins=True, prop=True, attr=True, ignore=ii.ignore+ii.plugin_ignore)
		if getattr(ii, '_prefix', None):
			_ = {k.lstrip(ii.prefix):v for k,v in d.items()}
		return d

	@property
	def d_flat(ii):
		return flat_any(ii.d)

# class Repo(PlugIn):
# 	class dist(PlugIn):
# 		Naive = Naive
# 		HFAccelerate = HFAccelerate
# 		SingleProcess = SingleProcess
	
# 	class logger(PlugIn):
# 		Logger = Wandb

# 	class resource(PlugIn):
# 		Niflheim = Niflheim
	
# 	class sweep(PlugIn):
# 		Optuna = Optuna

# 	class opt(PlugIn):
# 		Opt = OptBase

# 	class data(PlugIn):
# 		DataBase = DataBase

# 	class scheduler(PlugIn):
# 		SchedulerBase = SchedulerBase

# 	class paths(PlugIn):
# 		PathBase = PathsBase

# 	class model(PlugIn):
# 		ModelBase = ModelBase

def walk_ins_tree(
	ins: type, 
	k_update: str, 
	v_update: Any,
	v_ref = None,
):
	
	try:
		if hasattr(ins, k_update):
			v_ref = getattr(ins, k_update)
			v_update = type_me(v_update, v_ref)
			setattr(ins, k_update, v_update)
			print(f'updated {k_update}: \t {v_ref} ----> {v_update}')
			return True
		else:
			sub_ins = ins_to_dict(ins, sub_ins_ins=True)
			for v_ins in sub_ins.values():
				is_updated = walk_ins_tree(v_ins, k_update, v_update)
				if is_updated:
					return True
	except Exception as e:
		print(f'pyfig:walk_ins_tree k={k_update} v={v_update} v_ref={v_ref} ins={ins}')
	return False


def lo_ve(path:Path=None, data: dict=None):

	import torch
	""" loads anything you want (add other interfaces as needed) 

	1- str -> pathlib.Path
	2- get suffix (filetype)
	3- from filetype, decide encoding eg 'w' or 'wb'
	4- if data is present, decide to save
	5- from filetype, encoding, save or load, get or dump the data 
	"""

	file_interface_all = dict(
		pk = dict(
			rb = pk.load,
			wb = partial(pk.dump, protocol=pk.HIGHEST_PROTOCOL),
		),
		yaml = dict(
			r = yaml.load,
			w = yaml.dump,
		),
		json = dict(
			r = json.load,
			w = json.dump,
		),
		state = dict(
			r = torch.load,
			w = torch.save
		),
		cmd = dict(
			r = lambda f: ' '.join(f.readlines()),
			w = lambda x, f: f.writelines(x),
		),
		npy = dict(
			r = np.load,
			w = np.save,
		),
		npz = dict(
			r = np.load,
			w = np.savez_compressed,
		)
	)

	if isinstance(data, dict) and len(data)==0:
		print('lo_ve: data is empty dict- not dumping anything. Setting None and trying to load path.')
		data = None

	path = Path(path)
	if not path.suffix:
		path = path.with_suffix('.pk')
	file_type = path.suffix[1:]

	mode = 'r' if data is None else 'w'
	mode += 'b' if file_type in ['pk',] else ''
	interface = file_interface_all[file_type][mode]

	if 'r' in mode and not path.exists():
		return print('path: ', str(path), 'n\'existe pas- returning None')

	if 'state' in file_type:
		data = interface(data, path) if not data is None else interface(path)
	elif 'npz' in file_type:
		data = interface(path, **data) if not data is None else interface(path)
	else:
		with open(path, mode) as f:
			data = interface(data, f) if not data is None else interface(f)
	return data

def get_cls_d(ins: type, cls_k: list):
	return {k:getattr(ins.__class__, k) for k in cls_k}


def ins_to_dict( # docs:pyfig:ins_to_dict can only be used when sub_ins have been init 
	ins: type, 
	attr=False,
	sub_ins=False,  # always recursive
	sub_ins_ins=False,  # just the ins
	sub_cls=False,
	call=False,
	prop=False,
	ignore:list=None,
	flat:bool=False,
	debug: bool=False,
) -> dict:

	ignore = ignore or []

	cls_k = [k for k in dir(ins) if not k.startswith('_') and not k in ignore]
	cls_d = {k:getattr(ins.__class__, k) for k in cls_k}
	cls_prop_k = [k for k,v in cls_d.items() if isinstance(v, property)]
	cls_sub_cls_k = [k for k,v in cls_d.items() if isinstance(v, type)]
	cls_call_k = [k for k,v in cls_d.items() if (callable(v) and not (k in cls_sub_cls_k))]
	cls_attr_k = [k for k in cls_k if not k in (cls_prop_k + cls_sub_cls_k + cls_call_k)]

	ins_kv = []

	ins_sub_cls_or_ins = [(k, getattr(ins, k)) for k in cls_sub_cls_k]

	ins_kv += [(k,v) for (k,v) in ins_sub_cls_or_ins if isinstance(v, type) and sub_cls]

	for (k_ins, v_ins) in ins_sub_cls_or_ins:
		if not isinstance(v_ins, type):
			if sub_ins_ins: # just the ins
				ins_kv += [(k_ins, v_ins),]
			elif sub_ins: # recursive dict
				sub_ins_d: dict = ins_to_dict(v_ins, 
					attr=attr, sub_ins=sub_ins, prop=prop, 
					ignore=ignore, flat=flat)
				ins_kv += [(k_ins, sub_ins_d),]
	
	if prop: 
		ins_kv += [(k, getattr(ins, k)) for k in cls_prop_k]

	if call:
		ins_kv += [(k, getattr(ins, k)) for k in cls_call_k]

	if attr:
		ins_kv += [(k, getattr(ins, k)) for k in cls_attr_k]

	return flat_any(dict(ins_kv)) if flat else dict(ins_kv) 


