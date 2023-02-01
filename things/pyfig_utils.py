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
from functools import partial 
import pickle as pk
import yaml
import json
from functools import partial
import numpy as np

from .utils import dict_to_cmd, cmd_to_dict, dict_to_wandb, debug_dict
from .utils import mkdir, iterate_n_dir, gen_time_id, add_to_Path, dump, load
from .utils import get_cartesian_product, type_me, run_cmds, flat_any 

this_file_path = Path(__file__) 
hostname = os.environ['HOSTNAME']

class PlugIn:
	_p = None
	_prefix = None
	_static = True

	ignore = ['d', 'd_flat', 'ignore', 'plugin_ignore'] # applies to every plugin
	plugin_ignore: list = [] # can be used in the plugin versions

	def __init__(ii, parent=None):
		ii._p: PyfigBase
		ii._p: PyfigBase = parent

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
		n_b:        int         = None

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
		storage: 		Path = property(lambda _: 'sqlite:///'+str(_._p.exp_dir / 'hypam_opt.db'))
		sweep_name: 	str				= 'study' 
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
		
		dist_name: 	str			= None  # options: accelerate, naive
		sync_step:		int		= 5

		rank_env_name: 	str		= 'RANK'
		launch_cmd:		str		= property(lambda _: 
			lambda submit_i, cmd: None
		)

		rank: 			int 	= property(lambda _: int(os.environ.get(_.rank_env_name, '-1')))
		head: 			bool 	= property(lambda _: _.rank==0)
		n_process: 		int 	= property(lambda _: _._p.resource.n_gpu)

		gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_._gpu_id_cmd, silent=True)).split('.')[0])
		dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])
		
		_device: 		str		= 'cpu'
		_gpu_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
		_srun_cmd: 		str		= 'srun --gpus=1 --cpus-per-task=4 --ntasks=1 --exclusive --label --export=RANK={submit_i}'

		plugin_ignore: 	list	= ['launch_cmd']

		class dist_c(PlugIn):
			pass

		def sync(ii, v_d: dict, this_is_noop: bool=False) -> list[Any]:
			if this_is_noop or ii.n_process==1:
				return v_d
			else:
				return ii.dist_sync(v_d)

		def dist_sync(ii, v_d: dict) -> list[Any]:
			return v_d

		def backward(ii, loss, create_graph=False):
			loss.backward(create_graph=create_graph)

		def prepare(ii, *arg, **kw):
			return list(arg) + list(kw.values())

		def dist_set_seed(ii, seed):
			return seed
		
		def dist_set_device(ii, device):
			return device
		
		def dist_set_dtype(ii, dtype):
			return dtype

		def unwrap(ii, model):
			return model

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
	code_dir: 			Path    = property(lambda _: Path(_.exp_dir, 'code'))
	log_dir: 			Path    = property(lambda _: _.cluster_dir)


	_ignore_f = ['commit', 'pull', 'backward', 'accel', 'plugin_ignore']
	_ignore_c = ['parameters', 'scf']
	ignore: list = ['ignore', 'd', 'cmd', 'sub_ins', 'd_flat'] + _ignore_f + _ignore_c
	_group_i: int = 0
	_sub_ins: dict = {}
	step: int = -1
	zweep: str = ''

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

		update = flat_any((c_init or {})) | flat_any((other_arg or {})) | (sys_arg or {})

		### under construction ###
		from .distribute_utils import naive, hf_accel

		plugin_repo = dict(
			dist_name = dict(
				hf_accel=hf_accel,
				naive = naive,
			),
		)

		new_sub_cls = dict(filter(lambda kv: kv[0] in plugin_repo, update.items()))
		[update.pop(k) for k in new_sub_cls.keys()]
		for plugin, plugin_version in new_sub_cls.items():
			print('adding plugin:', plugin, plugin_version, '...')
			sub_cls = plugin_repo[plugin][plugin_version]
			plugin = plugin.split('_')[0]
			ii.init_sub_cls(sub_cls=sub_cls, name=plugin)

		### under construction ###

		print('update')
		pprint.pprint(update)
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

			tags = [str(s) for s in [*ii.mode.split('-'), ii.exp_id, ii._group_i]]
			
			if 'dark' in tags:
				print('pyfig:start:\n Going dark. wandb not initialised.')

			else:
				print(f'pyfig:wb: tags- {tags}')
				
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

				print(ii.wb.run_url)
				print('exp_dir: ', ii.exp_dir)
			

			ii.log(info=ii.cmd, path=ii.cluster_dir/'c.pyfig', group_exp=True)

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

				if ii.run_sweep or ii.wb.wb_sweep or ii.zweep:
					group_exp = not i==0
				else:
					group_exp = ii.group_exp

				ii.setup_exp_dir(group_exp= group_exp, force_new_id= True)
				base_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, 
							ignore=ii.ignore+['resource','dist_c','slurm_c','parameters'])
				run_d = base_d | run_d

				ii.debug_log([dict(os.environ.items()), run_d], ['log-submit_env.log', 'log-submit_d.log'])

				ii.resource.cluster_submit(run_d)

				print('log_group_url: \t\t\t',  ii.wb.run_url)
				print('exp_dir: \t\t\t', ii.exp_dir)
				print('exp_log: \t\t\t', ii._debug_paths['device_log_path'])

			ii.save_code_state()
			
			sys.exit('Exiting from submit.')

	def save_code_state(ii, exts = ['.py', '.ipynb', '.md']):
		import shutil
		shutil.copytree('things', ii.code_dir/'things')
		[shutil.copyfile(p, ii.code_dir/p.name) for p in ii.project_dir.iterdir() if p.suffix in exts]
		[shutil.copyfile(p, ii.code_dir/p.name) for p in ii.dump_dir.iterdir() if p.suffix in exts]

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
			c = [dict(),] 
			if ii.zweep: 
				ii.run_sweep = True
				# param-start-end-if anything  id-n_step-  i/a + c in range()
				# else categorical tuple
				zweep = ii.zweep.split('-')
				t = zweep[-1]
				c = [{zweep[0]: [i] if t=='list' else int(i)} for i in zweep[1:-1]] 
			return c
	
		if ii.wb.wb_sweep:
			param = ii.sweep.parameters
			sweep_keys = list(param.keys())

			n_sweep = 0
			for k, k_d in param.items():
				v = k_d.get('values', [])
				n_sweep += len(v)
	
			# n_sweep = len(get_cartesian_product(*(v for v in param))
			base_c = ins_to_dict(ii, sub_ins=True, attr=True, flat=True, ignore=ii.ignore+['parameters','head','exp_id'] + sweep_keys)
			base_cmd = dict_to_cmd(base_c, sep='=')
			base_sc = dict((k, dict(value=v)) for k,v in base_c.items())
   
			if ii.wb.wb_sweep:
				sweep_c = dict(
					command 	= ['python', '-u', '${program}', f'{base_c}', '${args}', '--exp_id=${exp_id}', ],
					program 	= str(Path(ii.run_name).absolute()),
					method  	= ii.sweep.sweep_method,
					parameters  = base_sc|param,
					# accel  = dict(type='local'),
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
				ii.log(d, path=ii.log_dir/name, group_exp=False)
			ii.log(d, path=ii.tmp_dir/name, group_exp=True)
			
	def partial(ii, f:Callable, args=None, **kw):
		d = flat_any(args if args else {}) | ii.d_flat | (kw or {})
		d_k = inspect.signature(f.__init__).parameters.keys()
		d = {k:v for k,v in d.items() if k in d_k} 
		return f(**d)

	def update(ii, _arg: dict=None, c_update: dict=None, **kw):
		print('\npyfig:update')

		c_update = (_arg or {}) | (c_update or {}) | (kw or {})
		
		arg = flat_any(c_update)
		
		c_keys = list(ii.d_flat.keys())
		
		arg = dict(filter(lambda kv: kv[0] in c_keys, arg.items()))

		for k_update, v_update in deepcopy(arg).items():
			is_updated = walk_ins_tree(ii, k_update, v_update)
			if not is_updated:

				if k_update=='dtype': # !!! pyfig:fix: Special case, generalify
					ii.dtype = v_update
					print('dtype: --- ', v_update)
				else:
					print(f'not updated: k={k_update} v={v_update} type={type(v_update)}')

		not_arg = dict(filter(lambda kv: kv[0] not in c_keys, arg.items()))
		debug_dict(msg='update:not = \n', not_arg=not_arg)
		
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

	def set_dtype(ii, dtype = None):
		import torch
		ii.dtype = torch.float64
		print('setting default dtype: ', print(ii.dtype))
		ii.dtype = dtype or ii.dtype
		print('setting default dtype: ', dtype)
		torch.set_default_dtype(ii.dtype)
		ii.dtype = torch.randn((1,)).dtype
		print('pyfig: dtype is ', ii.dtype)
		return ii.dtype

	def set_device(ii, device: str=None):
		import torch
		if device:
			ii.device = device
		elif not torch.cuda.is_available(): 
			print('Running on cpu', ii.dist.dist_id)
			ii.device = 'cpu'
		elif ii.dist.dist_name=='hf_accel':
			print('Getting device from dist plugin', ii.dist.dist_name)
			print(dir(ii.dist))
			ii.device = ii.dist.dist_set_device(ii.device)
		else:
			device_int = torch.cuda.current_device()
			torch_n_device = torch.cuda.device_count()
			cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
			print('CUDA_VISIBLE_DEVICES:', cuda_visible_devices)
			ii.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print('pyfig: Device is ', ii.device)
		return ii.device

	def set_seed(ii, seed=None):
		print('\npyfig:set_seed:')

		if callable(getattr(ii.dist, 'dist_set_seed')):
			print(f'dist plugin setting seed')
			seed = ii.dist.dist_set_seed(ii.seed)
		else:
			import torch
			print(f'pyfig setting seed')
			ii.seed = ii.seed + ii.dist.gpu_i
			torch.random.manual_seed(ii.seed)
		print('pyfig:set_seed seed=', ii.seed)
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
		global accel
		global DataLoader

		from accelerate import accel
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