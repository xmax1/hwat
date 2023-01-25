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

from torch import nn

this_dir = Path(__file__).parent
hostname = os.environ['HOSTNAME']

class PlugIn:
	_p = None
	_prefix = None
	ignore = ['d', 'd_flat', 'ignore', 'plugin_ignore'] # applies to every plugin
	plugin_ignore: list = [] # can be used in the plugin versions

	def __init__(ii, parent=None):
		ii._p: PyfigBase
		ii._p: PyfigBase = parent
		ii.init_sub_cls()
  
	def init_sub_cls(ii,) -> dict:
		sub_cls = ins_to_dict(ii, sub_cls=True)
		for sub_k, sub_v in sub_cls.items():
			sub_ins = sub_v(parent=ii)
			setattr(ii, sub_k, sub_ins)

	@property
	def d(ii):
		d = ins_to_dict(ii, sub_ins=True, prop=True, attr=True, ignore=ii.ignore+ii.plugin_ignore)
		if getattr(ii, '_prefix', None):
			_ = {k.lstrip(ii._prefix):v for k,v in d.items()}
		return d

	@property
	def d_flat(ii):
		return flat_any(ii.d)

class Param(PlugIn): 
	values: list = None
	domain: tuple = None
	dtype: type = None
	log: bool = False
	step_size: float|int = None
	sample: str = None # docs:Param:sample from ('uniform', )
	condition: list = None

	def __init__(ii, 
		values=None, 
		domain=None, 
		dtype=None, 
		log=None, 
		step_size=None, 
		sample=None, 
		condition=None, 
		parent=None

	) -> None: # docs:Param:init needed so can use kw arg to init
		super().__init__(parent=parent)

		ii.values = values
		ii.domain = domain
		ii.dtype = dtype
		ii.log = log
		ii.sample = sample
		ii.step_size = step_size
		ii.condition = condition

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
	run_sweep:      	bool    = False
	
	seed:           	int   	= 0
	dtype:          	str   	= ''
	device: 		int|str 	= ''

	n_step:         	int   	= 0
	n_eval_step:        int   	= 0
	n_pre_step:    		int   	= 0

	log_metric_step:	int   	= 0
	log_state_step: 	int   	= 0

	lo_ve_path:			str 	= ''

	group_i: 			int 	= property(lambda _: _._group_i)
	
	class data(PlugIn):
		system: 	str = ''
		n_b:        int         = 0

		charge:     int         = 0
		spin:       int         = 0
		a:          np.ndarray  = np.array([[0.0, 0.0, 0.0],])
		a_z:        np.ndarray  = np.array([4.,])

		n_corr:     int         = 0
		n_equil_step:		int	= 0
		acc_target: int         = 0

		n_e:        int         = property(lambda _: int(sum(_.a_z)))
		n_u:        int         = property(lambda _: (_.spin + _.n_e)//2)
		n_d:        int         = property(lambda _: _.n_e - _.n_u)

	class model(PlugIn):
		with_sign:      bool    = False
		terms_s_emb:    list    = ['ra', 'ra_len']
		terms_p_emb:    list    = ['rr', 'rr_len']
		ke_method:      str     = 'vjp'
		n_sv:           int     = 32
		n_pv:           int     = 16
		n_fb:           int     = 2
		n_det:          int     = 1
  
		n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)

	class opt(PlugIn):
		opt_name: 		str		= None
		lr:  			float 	= None
		betas:			list	= None
		eps: 			float 	= None
		weight_decay: 	float 	= None
		hessian_power: 	float 	= None
  
	class sweep(PlugIn):
		storage: 		Path = property(lambda _: 'sqlite:///' + str(_._p.exp_dir / 'hypam_opt.db'))
		sweep_name: 	str				= '' 
		sweep_method: 	str				= '' # wb name of alg: grid,bayes, ... 
		parameters: 	dict[Param]		= {}
		n_trials: 		int				= 10
  
	class wb(PlugIn):
		run = None
		job_type:		str		= ''		
		wb_mode: 		str		= ''
		wb_sweep: 		bool	= False
		sweep_id: 		str 	= ''
		wb_run_path:	str 	= ''

		entity:			str		= property(lambda _: _._p.project)
		program: 		Path	= property(lambda _: Path( _._p.project_dir, _._p.run_name))
		sweep_path_id:  str     = property(lambda _: f'{_.entity}/{_._p.project}/{_.sweep_id}')
		wb_type: 		str		= property(lambda _: _.wb_sweep*f'sweeps/{_.sweep_id}' or f'groups/{_._p.exp_name}/workspace') #  or f'runs/{_._p.exp_id} _._p.group_exp*f'groups' or 'runs') # or _._p.run_sweep*f'groups/{_._p.exp_name}'
		run_url: 		str		= property(lambda _: f'https://wandb.ai/{_.entity}/{_._p.project}/{_.wb_type}')
		_wb_agent: 		bool	= False


	class dist(PlugIn):
		head:			bool	= True
		dist_method: 	str		= 'pyfig'  # options: accelerate
		sync_step:		int		= None

		dist_set_seed: Callable = None
		dist_set_device: Callable = None
		dist_set_dtype: Callable = None

		_device: 		str		= 'cpu'
		_srun_cmd: 		str		= 'srun --gpus=1 --cpus-per-task=4 --ntasks=1 --exclusive --label '
		_gpu_id_cmd:		str	= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'

		_launch_cmd:	str  	= property(lambda _: f'{_._srun_cmd} python -u {_._p.run_name} ')
		rank: 			bool 	= property(lambda _: os.environ.get('RANK', '0'))
		gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_._gpu_id_cmd, silent=True)).split('.')[0])
		dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])

		class dist_c(PlugIn):
			pass

		def __init__(ii, parent=None):
			super().__init__(parent=parent)

			def debug_sync(v_d: dict, step: int):
				v_sync = ii.sync(v_d, step)
				return v_sync
				
			ii.sync = debug_sync

		@torch.no_grad()
		def sync(ii, step: int, v_d: dict) -> dict:
			return {}

		def backward(ii, loss: torch.Tensor):
			opt_is_adahess = ii._p.opt.opt_name.lower()=='AdaHessian'.lower()
			loss.backward(create_graph=opt_is_adahess)

		def prepare(ii, *arg, **kw):
			return list(arg) + list(kw.values())
 

	class resource(PlugIn):
		submit: 		bool	= False
		cluster_submit: Callable= None
		script:			Callable= None
		device_log_path:Callable= None

	home:				Path	= Path().home()
	project_dir:        Path    = property(lambda _: _.home / 'projects' / _.project)
	dump_dir:           Path    = property(lambda _: Path('dump'))
	tmp_dir:            Path	= property(lambda _: Path(_.dump_dir,'tmp'))
	exp_dir:        	Path	= property(lambda _: Path(_.dump_exp_dir, _.exp_name, _.exp_id))
	dump_exp_dir:       Path	= property(lambda _: Path(_.dump_dir, 'exp'))
	cluster_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'cluster'))
	exchange_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'exchange'))
	profile_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'profile'))
	state_dir: 			Path    = property(lambda _: Path(_.exp_dir, 'state'))
	exp_data_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'exp_data'))
	log_dir: 			Path    = property(lambda _: _.cluster_dir)

	_ignore_f = ['commit', 'pull', 'backward', 'controller', 'plugin_ignore']
	_ignore_c = ['sweep',]
	ignore: list = ['ignore', 'd', 'cmd', 'sub_ins', 'd_flat'] + _ignore_f + _ignore_c
	_group_i: int = 0
	_sub_ins: dict = {}
	step: int = -1

	def __init__(ii, 
		notebook:bool=False,  # removes sys_arg for notebooks
		sweep: dict={},  # special properties for config update so is separated
		c_init: dict|str|Path={},  # args specificall  
		post_init_arg: dict={},
		**other_arg):     

		ii.init_sub_cls()

		for k,v in (sweep or {}).items():
			setattr(ii.sweep, k, v)

		if not notebook:
			ref_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, ignore=ii.ignore)
			sys_arg = cmd_to_dict(sys.argv[1:], ref_d)

		update = flat_any((c_init or {})) | flat_any((other_arg or {})) | (sys_arg or {})


		### under construction ###
		from .distribute_utils import naive, hf_accelerate
		plugin_repo = dict(
			dist = dict(
				hf_accelerate=hf_accelerate,
				naive = naive,
			),
		)
		new_sub_cls = dict(filter(lambda kv: kv[0] in ii._sub_ins.keys(), update.items()))
		[update.pop(k) for k in new_sub_cls.keys()]
		for plugin, plugin_version in new_sub_cls.items():
			sub_cls = plugin_repo[plugin][plugin_version]
			ii.init_sub_cls(sub_cls=sub_cls, name=plugin)
		### under construction ###
		
		ii.update(update)

		if ii.debug:
			os.environ['debug'] = 'True'

		ii.debug_log([sys_arg, dict(os.environ.items()), ii.d], ['log_sys_arg.log', 'log_env_run.log', 'log_d.log'])

	def start(ii, dark=False):

		if ii.dist.head and int(ii.dist.rank)==0 and not dark:
			assert ii.resource.submit == False

			print('start:wb:init creating the group')
			ii.setup_exp_dir(group_exp= ii.group_exp, force_new_id= False)

			ii._group_i += 1
		
			ii.run_id = ii.exp_id + '.' + ii.mode + '.' + str(ii._group_i)
			ii.wb.wb_run_path = f'{ii.wb.entity}/{ii.project}/{ii.run_id}'  # no / no : - in mode _ in names try \ + | .
			# ii.run_id = '.'.join(ii.run_id.split('.'))  
			
			print('start:wb:init:exp_dir = \n ***', ii.exp_dir, '***')
			print('start:wb:init:wb_run_path = \n ***', ii.wb.wb_run_path, '***')
			print('start:wb:init:run_id = \n ***', ii.run_id, '***')
			ii.setup_exp_dir(group_exp= False, force_new_id= False)

			tags = ii.mode.split('-')
			

			if 'dark' in tags:
				print('pyfig:start:\n Going dark. wandb not initialised.')

			else:
				print(f'pyfig:wb: tags- {tags}')
				# os.environ['WANDB_START_METHOD'] = "thread"
				ii.wb.run = wandb.init(
					project     = ii.project, 
					group		= ii.exp_name,
					dir         = ii.exp_data_dir,
					entity      = ii.wb.entity,	
					mode        = ii.wb.wb_mode,
					config      = dict_to_wandb(ii.d_flat),
					id			= ii.run_id,
					tags 		= tags,
					reinit 		= not (ii.wb.run is None)
				)

	def end(ii):
		try: 
			ii.wb.run.finish()
		except:
			pass

	def run_local_or_submit(ii):

		if ii.resource.submit:
			print('submitting to cluster')
			ii.resource.submit = False # docs:submit

			run_or_sweep_d = ii.get_run_or_sweep_d()

			for i, run_d in enumerate(run_or_sweep_d):

				if ii.run_sweep or ii.wb.wb_sweep:
					group_exp = not i==0
				else:
					group_exp = ii.group_exp

				ii.setup_exp_dir(group_exp= group_exp, force_new_id= True)
				base_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, 
							ignore=ii.ignore+['sweep','resource','dist_c','slurm_c'])
				run_d = base_d | run_d

				ii.debug_log([dict(os.environ.items()), run_d], ['log-submit_env.log', 'log-submit_d.log'])

				ii.resource.cluster_submit(run_d)

				print(ii.wb.run_url)
			
			sys.exit('Exiting from submit.')

	@staticmethod
	def pr(d: dict):
		pprint.pprint(d)
		return d

	def make_a_note(ii, path: str|Path, note: str|list):
		note = [note,] if isinstance(note, str) else note
		with open(path, 'w') as f:
			f.writelines(note)

	@property
	def _paths(ii):
		""" check """
		path_filter = lambda item: any([p in item[0] for p in ['path', 'dir']])
		paths = dict(filter(path_filter, ii.d.items()))
		return ii.pr(paths) if ii.debug else paths

	@property
	def _debug_paths(ii):
		return dict(device_log_path=ii.resource.device_log_path(rank=0))

	@property
	def cmd(ii):
		return dict_to_cmd(ii.d)

	@property
	def d(ii):
		return ins_to_dict(ii, sub_ins=True, prop=True, attr=True, ignore=ii.ignore)

	@property
	def d_flat(ii):
		return flat_any(ii.d)

	def init_sub_cls(ii, sub_cls: type=None, name: str=None) -> dict:
		if name and not (sub_cls is None):
			sub_ins = sub_cls(parent=ii)
			setattr(ii, name, sub_ins)
			ii._sub_ins[name] = sub_ins
		else:
			sub_cls = ins_to_dict(ii, sub_cls=True)
			for sub_k, sub_cls_i in sub_cls.items():
				sub_ins = sub_cls_i(parent=ii)
				setattr(ii, sub_k, sub_ins)
				ii._sub_ins[sub_k] = sub_ins

	def setup_exp_dir(ii, group_exp=False, force_new_id=False):

		if (not ii.exp_id) or force_new_id:
			ii.exp_id = gen_time_id(7)

			exp_name = ii.exp_name or 'junk'
			group_exp = group_exp or ('~' in exp_name)
			sweep_dir = 'sweep'*ii.run_sweep
			exp_group_dir = Path(ii.dump_exp_dir, sweep_dir, exp_name)
			exp_group_dir = iterate_n_dir(exp_group_dir, group_exp=group_exp) # pyfig:setup_exp_dir does not append -{i} if group allowed
			ii.exp_name = exp_group_dir.name

		print('exp_dir: ', ii.exp_dir) 
		[mkdir(p) for _, p in ii._paths.items()]
	
	def get_run_or_sweep_d(ii,):
		
		if not (ii.run_sweep or ii.wb.wb_sweep):
			""" single run takes c from base in submit loop """
			return [dict(),] 
	
		if ii.wb.wb_sweep:
			param = ii.sweep.parameters
			sweep_keys = list(param.keys())

			n_sweep = 0
			for k, k_d in param.items():
				v = k_d.get('values', [])
				n_sweep += len(v)
	
			# n_sweep = len(get_cartesian_product(*(v for v in param))
			base_c = ins_to_dict(ii, sub_ins=True, attr=True, flat=True, ignore=ii.ignore+['sweep', 'head', 'exp_id'] + sweep_keys)
			base_cmd = dict_to_cmd(base_c, sep='=')
			base_sc = dict((k, dict(value=v)) for k,v in base_c.items())
   
			if ii.wb.wb_sweep:
				sweep_c = dict(
					command 	= ['python', '-u', '${program}', f'{base_c}', '${args}', '--exp_id=${exp_id}', ],
					program 	= str(Path(ii.run_name).absolute()),
					method  	= ii.sweep.sweep_method,
					parameters  = base_sc|param,
					controller  = dict(type='local'),
				)

				os.environ['WANDB_PROJECT'] = 'hwat'
				os.environ['WANDB_ENTITY'] = 'hwat'

				ii.wb.sweep_id = wandb.sweep(
					sweep_c, 
					project = ii.project,
					entity  = ii.wb.entity
				)
				api = wandb.Api()
				sweep = api.sweep(str(ii.wb.sweep_path_id))
				n_sweep_exp = sweep.expected_run_count
				print(f"EXPECTED RUN COUNT = {n_sweep_exp}")
				print(f"EXPECTED RUN COUNT = {n_sweep}")
				print(ii.project, ii.wb.entity, Path(ii.run_name).absolute())

			return [dict() for _ in range(n_sweep)]
 
		d = ii.sweep.d
		sweep_keys = list(d['parameters'].keys())
		sweep_vals = [v.get('values', []) for v in d['parameters'].values()]
		sweep_vals = get_cartesian_product(*sweep_vals)
		print(f'### sweep over {sweep_keys} ({len(sweep_vals)} total) ###')
		return [{k:v for k,v in zip(sweep_keys, v_set)} for v_set in sweep_vals]

	def debug_log(ii, d_all:list, name_all: list):
		for d, name in zip(d_all, name_all):
			if Path(ii.exp_dir).is_dir():
				ii.log(d, path=ii.log_dir/name)
			ii.log(d, path=ii.tmp_dir/name)
			
	def partial(ii, f:Callable, args=None, **kw):
		d = flat_any(args if args else {}) | ii.d_flat | (kw or {})
		d_k = inspect.signature(f.__init__).parameters.keys()
		d = {k:v for k,v in d.items() if k in d_k} 
		return f(**d)

	def update(ii, _arg: dict=None, c_update: dict=None, **kw):
		c_update = (_arg or {}) | (c_update or {}) | (kw or {})
		arg = flat_any(c_update)
		c_keys = list(ii.d_flat.keys())
		arg = dict(filter(lambda kv: kv[0] in c_keys, arg.items()))

		for k_update, v_update in deepcopy(arg).items():
			is_updated = walk_ins_tree(ii, k_update, v_update)
			if not is_updated:
				print(f'not updated: k={k_update} v={v_update} type={type(v_update)}')

		not_arg = dict(filter(lambda kv: kv[0] not in c_keys, arg.items()))
		debug_dict(msg='update:not = \n', not_arg=not_arg)
		
	@staticmethod
	def log(info: dict|str, path: Path):
		mkdir(path)
		info = pprint.pformat(info)
		with open(path, 'w') as f:
			f.writelines(info)

	def to(ii, framework='torch'):
		base_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, ignore=ii.ignore+['sweep'])
		d = {k:v for k,v in base_d.items() if isinstance(v, (np.ndarray, np.generic))}
		if 'torch' in framework.lower():
			d = {k:torch.tensor(v, requires_grad=False).to(device=ii.device, dtype=ii.dtype) for k,v in d.items()}
		if 'numpy' in framework.lower():
			d = {k:v.detach().cpu().numpy() for k,v in d.items() if isinstance(v, torch.Tensor)}
		ii.update(d)

	def set_dtype(ii, dtype=torch.DoubleTensor):
		print('setting default dtype: ', dtype)
		ii.dtype = dtype or ii.dtype
		torch.set_default_tensor_type(ii.dtype) 
		ii.dtype = torch.randn((1,)).dtype
		print('pyfig: dtype is ', ii.dtype)
		return ii.dtype

	def set_device(ii, device: str=None):
		ii.device = 'cpu'
		if device: 
			ii.device = device
		elif not getattr(ii.dist, 'dist_set_device') is None:
			ii.device = ii.dist.dist_set_device(ii.device)
		else:
			device_int = torch.cuda.current_device()
			torch_n_device = torch.cuda.device_count()
			cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
			ii.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print('pyfig: Device is ', ii.device)
		return ii.device

	def set_seed(ii, seed=None):
		ii.seed = seed or ii.seed
		if not getattr(ii.dist, 'dist_set_seed') is None:
			print(f'plugin:dist: \"dist plugin setting seed.\"')
			ii.seed = ii.dist.dist_set_seed(ii.seed)
		else:
			print(f'pyfig: \"Setting seed {ii.seed} w torch manually.\"')
			torch.random.manual_seed(ii.seed)
		print('pyfig: Seed is ', ii.seed)
		return ii.seed

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
	path = Path(path)
	if not path.suffix:
		path = path.with_suffix('.pk')
	file_type = path.suffix[1:]

	mode = 'r' if data is None else 'w'
	mode += 'b' if file_type in ['pk',] else ''
	interface = file_interface_all[file_type][mode]

	if 'r' in mode and not path.exists():
		return print('path: ', str(path), 'n\'existe pas- returning None')

	if 'state' in file_type or 'npz' in file_type:
		data = interface(data, path) if not data is None else interface(path)
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
	sub_ins_tag:str='_p',
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
					ignore=ignore, flat=flat, sub_ins_tag=sub_ins_tag
				)
				ins_kv += [(k_ins, sub_ins_d),]
	
	if prop: 
		ins_kv += [(k, getattr(ins, k)) for k in cls_prop_k]

	if call:
		ins_kv += [(k, getattr(ins, k)) for k in cls_call_k]

	if attr:
		ins_kv += [(k, getattr(ins, k)) for k in cls_attr_k]

	return flat_any(dict(ins_kv)) if flat else dict(ins_kv) 


class Metrix:
	step: 		   int 	 = 0
	t0: 		   float = time()
	max_mem_alloc: float = None
	t_per_it: 	   float = None
	opt_obj: 	   float = None
	opt_obj_all:    list = None
	eval_keys: 	   list  = None
	
	log: 			dict = None

	exp_stats: 		list = ['max_mem_alloc', 't_per_it', 'opt_obj', 'opt_obj_all']
	source: 		str  = 'exp_stats/'

	def __init__(ii, eval_keys: list=None):
		
		ii.eval_keys = eval_keys
		ii.opt_obj_all = []

		torch.cuda.reset_peak_memory_stats()

	def tick(ii, step: int, opt_obj: float=None, **kw) -> dict:
		dstep = step - ii.step 

		t1 = time()
		ii.t_per_it = (t1 - ii.t0) / float(dstep)
		ii.t0 = time()

		ii.max_mem_alloc = torch.cuda.max_memory_allocated() // 1024 // 1024
		torch.cuda.reset_peak_memory_stats()

		ii.opt_obj = opt_obj
		ii.opt_obj_all += [ii.opt_obj,]

		return dict(exp_stats={k: getattr(ii, k) for k in ii.exp_stats})
	
	def to_dict(ii):
		return {k: getattr(ii, k) for k in ii.exp_stats}


# prefix components:
draw_space =  '    '
draw_branch = '│   '
# pointers:
draw_tee =    '├── '
draw_last =   '└── '

def recurse_tree(dir_path: Path, prefix: str=''):
	"""A recursive generator, given a directory Path object
	will yield a visual tree structure line by line
	with each line prefixed by the same characters
	"""    
	contents = list(dir_path.iterdir())
	# contents each get pointers that are ├── with a final └── :
	pointers = [draw_tee] * (len(contents) - 1) + [draw_last]
	for pointer, path in zip(pointers, contents):
		yield prefix + pointer + path.name
		if path.is_dir(): # extend the prefix and recurse:
			extension = draw_branch if pointer == draw_tee else draw_space 
			# i.e. space because last, └── , above so no more |
			yield from recurse_tree(path, prefix=prefix+extension)
	

def tree(
	dir_path: Path, 
	level: int=4, 
	limit_to_directories: bool=False,
	length_limit: int=10
):
	"""Given a directory Path object print a visual tree structure"""
	dir_path = Path(dir_path) # accept string coerceable to Path
	files = 0
	directories = 0
	def inner(dir_path: Path, prefix: str='', level=-1):
		nonlocal files, directories
		if not level: 
			return # 0, stop iterating
		if limit_to_directories:
			contents = [d for d in dir_path.iterdir() if d.is_dir()]
		else: 
			contents = list(dir_path.iterdir())
		pointers = [draw_tee] * (len(contents) - 1) + [draw_last]
		for pointer, path in zip(pointers, contents):
			if path.is_dir():
				yield prefix + pointer + path.name
				directories += 1
				extension = draw_branch if pointer == draw_tee else draw_space 
				yield from inner(path, prefix=prefix+extension, level=level-1)
			elif not limit_to_directories:
				yield prefix + pointer + path.name
				files += 1
	print(dir_path.name)
	iterator = inner(dir_path, level=level)
	for line in islice(iterator, length_limit):
		print(line)
	if next(iterator, None):
		print(f'... length_limit, {length_limit}, reached, counted:')
	print(f'\n{directories} directories' + (f', {files} files' if files else ''))

def draw_tree():
	recurse_tree(Path.home() / 'pyscratch')
 
 
""" 

class dist_pyfig(PlugIn):
 
	def __init__(ii, parent=None):
		super().__init__(parent)
		global Accelerator
		global DataLoader

		from accelerate import Accelerator
		from torch.utils.data import DataLoader
 
	# class git(PlugIn):
	# 	branch:     str     = 'main'
	# 	remote:     str     = 'origin' 

	# 	_commit_id_cmd:	str 	= 'git log --pretty=format:%h -n 1'
	# 	commit_id:   	list	= property(lambda _: run_cmds(_._commit_id_cmd, cwd=_._p.project_dir, silent=True))
		# commit_cmd:	str     = 'git commit -a -m "run"' # !NB no spaces in msg 
		# commit: 		list	= property(lambda _: run_cmds(_.commit_cmd, cwd=_.project_dir)) 
		# pull_cmd:		str 	= ['git fetch --all', 'git reset --hard origin/main']
		# pull:   		list	= property(lambda _: run_cmds(_.pull_cmd, cwd=_.project_dir))
  
  
def ins_to_dict( # docs:pyfig:ins_to_dict can only be used when sub_ins have been init 
	ins: type, 
	attr=False,
	sub_ins=False,
	sub_cls=False,
	call=False,
	prop=False,
	ignore:list=None,
	flat:bool=False,
	sub_ins_tag:str='_p',
) -> dict:

	ignore = ignore or []

	# ins_d = {k:getattr(ins, k) for k in dir(ins)}
	cls_k = (k for k in dir(ins) if not k.startswith('_') and not k in ignore)
	cls_d = get_cls_d(ins)

	
	sub_ins_kv, prop_kv, call_kv, attr_kv, sub_cls_kv = [], [], [], [], []
	for cls_k, cls_v in cls_d.items():

		if isinstance(v, type):
			sub_cls_kv += [(k,v),]
   
		elif hasattr(v, sub_ins_tag):
			sub_ins: dict = ins_to_dict(v, 
				attr=attr, sub_ins=sub_ins, prop=prop, 
				ignore=ignore, flat=flat, sub_ins_tag=sub_ins_tag
			)
			sub_ins_kv += list(sub_ins.items())
   
		elif prop: # docs:pyfig:sub_cls:issues
			if isinstance(getattr(ins.__class__, k), property):
				prop_kv += [(k,v),]
	
		elif call and (callable(v) and not hasattr(v, sub_ins_tag)):
			call_kv += [(k,v),]
   
		elif attr:
			attr_kv += [(k,v),]

		else:
			pass

	# final_d = dict(prop*prop_kv + sub_ins*sub_ins_kv + attr*attr_kv + sub_cls*sub_cls_kv) # + call*call_kv one day but not needed
	final_d = dict(prop_kv + sub_ins_kv + attr_kv + sub_cls_kv) # + call*call_kv one day but not needed
	return flat_any(final_d) if flat else final_d # docs:python:flat:q? parameter to flat dict in levels?
# server local https://github.com/wandb/wandb/issues/4586
			# launch_cmd = 'srun --gpus=1 --ntasks=1 --exclusive --label  '
			# body += [f'{launch_cmd} python -u {ii._p.run_name} {dict_to_cmd(job)} 1> {ii.device_log_path(rank=0)} 2>&1 & ']
				# srun --nodes=1 --mpi=cray_shasta --gpus=1 --cpus-per-task=8 --ntasks=1 -vvvvv --label --exclusive bash -c &


		class _Body:
			lines = []
			def __iadd__(ii, l):
				ii.lines += [l]
			def __repr__(self) -> str:
				pass
			def __str__(ii) -> str:
				return '\n'.join(ii.lines)

		try:
			_body = _Body()
			_body += 'x'
			print(_body)
		except Exception as e:
			print(e)

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
			r = torch.save,
			w = torch.load
		),
		cmd = dict(
			r = lambda f: ' '.join(f.readlines()),
			w = lambda x, f: f.writelines(x),
		),
		npy = dict(
			rb = np.load,
			wb = np.save,
		)
	) 
"""