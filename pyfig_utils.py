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
from utils import get_max_n_from_filename

from utils import dict_to_cmd, cmd_to_dict, dict_to_wandb, debug_dict
from utils import mkdir, iterate_n_dir, gen_time_id, add_to_Path, dump, load
from utils import get_cartesian_product, type_me, run_cmds, flat_any 
from dump.systems import systems

from torch import nn

this_dir = Path(__file__).parent
hostname = os.environ['HOSTNAME']

class Sub:
	_p = None
	_sub_ins_tag = '_p'
	ignore = ['d', 'd_flat', 'ignore']

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
		return ins_to_dict(ii, sub_ins=True, prop=True, attr=True, ignore=ii.ignore)

	@property
	def d_flat(ii):
		return flat_any(ii.d)

class Param(Sub): 
	# docs:todo all wb sweep structure
	values: list = None
	domain: tuple = None
	dtype: type = None
	log: bool = False
	step_size: float|int = None
	sample: str = None # docs:Param:sample from ('uniform', )

	def __init__(ii, values=None, domain=None, dtype=None, log=None, step_size=None, sample=None, parent=None) -> None: # docs:Param:init needed so can use kw arg to init
		super().__init__(parent=parent)
		ii.values = values
		ii.domain = domain
		ii.dtype = dtype
		ii.log = log
		ii.sample = sample
		ii.step_size = step_size

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

	n_step:         	int   	= 0
	n_eval_step:        int   	= 0
	n_pre_step:    		int   	= 0

	log_metric_step:	int   	= 0
	log_state_step: 	int   	= 0

	lo_ve_path:			str 	= ''

	group_i: 			int 	= property(lambda _: _._group_i)
	
	class data(Sub):
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

	class model(Sub):
		with_sign:      bool    = False
		terms_s_emb:    list    = ['ra', 'ra_len']
		terms_p_emb:    list    = ['rr', 'rr_len']
		ke_method:      str     = 'vjp'
		n_sv:           int     = 32
		n_pv:           int     = 16
		n_fb:           int     = 2
		n_det:          int     = 1
  
		n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)

	class opt(Sub):
		opt_name: 		str		= None
		lr:  			float 	= None
		betas:			list	= None
		eps: 			float 	= None
		weight_decay: 	float 	= None
		hessian_power: 	float 	= None
  
	class sweep(Sub):
		storage: 		Path = property(lambda _: 'sqlite:///' + str(_._p.exp_dir / 'hypam_opt.db'))
		sweep_name: 	str				= '' 
		sweep_method: 	str				= '' # wb name of alg: grid,bayes, ... 
		parameters: 	dict[Param]		= {}
		n_trials: 		int				= 10
  
	class wb(Sub):
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
  
	class distribute(Sub):
		head:			bool	= True
		device: 		str		= 'cpu'
		dist_method: 	str		= 'pyfig'  # options: accelerate
		sync_step:		int		= 5
  
		_srun_cmd: 		str		= 'srun --gpus=1 --cpus-per-task=4 --ntasks=1 --exclusive --label '
		_gpu_id_cmd:		str	= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'

		_launch_cmd:	str  	= property(lambda _: f'{_._srun_cmd} python -u {_._p.run_name} ')
		rank: 			bool 	= property(lambda _: os.environ.get('RANK', '0'))
		gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_._gpu_id_cmd, silent=True)).split('.')[0])
		dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])

		class dist_c(Sub):
			pass

		def __init__(ii, parent=None):
			super().__init__(parent)

			def debug_sync(v_d: dict, step: int):
				if ((step/ii.sync_step)==1) and ii._p.debug:
					[print(k, v.shape) for k,v in v_d.items()]
				v_sync = ii.sync(v_d, step)
				if ((step/ii.sync_step)==1) and ii._p.debug:
					[print(k, v.shape) for k,v in v_sync.items()]
				return v_sync
				
			ii.sync = debug_sync

		def sync(ii, step: int, v_d: dict) -> dict:
			v_path = (ii._p.exchange_dir / f'{step}_{ii._p.distribute.dist_id}').with_suffix('.pk')
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
			
			if ii._p.distribute.head:

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

		def backward(ii, loss: torch.Tensor):
			opt_is_adahess = ii._p.opt.opt_name.lower() == 'AdaHessian'.lower()
			loss.backward(create_graph=opt_is_adahess)

		def set_device(ii, device=None):
			ii.device = 'cpu'
			is_cuda = torch.cuda.is_available()
			if is_cuda:
				torch_curr_device = torch.cuda.current_device()
				torch_n_device = torch.cuda.device_count()
				cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
				ii.device = 'cuda' # default is cpu
			return ii.device

		def set_seed(ii, seed=None):
			print('setting seed w torch manually ' )
			torch.random.manual_seed(seed or ii._p.seed)

		def prepare(ii, *arg, **kw):
			return list(arg) + list(kw.values())
 
	class resource(Sub):
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

	_ignore_f = ['commit', 'pull', 'backward']
	_ignore_c = ['sweep',]
	ignore: list = ['ignore', 'd', 'cmd', 'sub_ins', 'd_flat'] + _ignore_f + _ignore_c
	_sub_ins_tag: str = '_p'
	_group_i: int = 0

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
		ii.update(update)

		if ii.debug:
			os.environ['debug'] = 'True'

		ii.debug_log([sys_arg, dict(os.environ.items()), ii.d], ['log_sys_arg.log', 'log_env_run.log', 'log_d.log'])

	def start(ii):

		if ii.distribute.head and int(ii.distribute.rank)==0:
			assert ii.resource.submit == False

			print('start:wb:init creating the group')
			ii.setup_exp_dir(group_exp= ii.group_exp, force_new_id= False)

			ii._group_i += 1
			if ii._group_i==1:
				ii.run_id = ii.exp_id
			
			ii.run_id = ii.exp_id + '.' + ii.mode + '.' + str(ii._group_i)
			ii.wb.wb_run_path = f'{ii.wb.entity}/{ii.project}/{ii.run_id}'
			ii.run_id = '.'.join(ii.run_id.split('.'))  # no / no : - in mode _ in names try \ + | .
			
			print('start:wb:init:exp_dir = \n ***', ii.exp_dir, '***')
			print('start:wb:init:wb_run_path = \n ***', ii.wb.wb_run_path, '***')
			print('start:wb:init:run_id = \n ***', ii.run_id, '***')
			ii.setup_exp_dir(group_exp= False, force_new_id= False)

			ii.wb.run = wandb.init(
				project     = ii.project, 
				group		= ii.exp_name,
				dir         = ii.exp_data_dir,
				entity      = ii.wb.entity,	
				mode        = ii.wb.wb_mode,
				config      = dict_to_wandb(ii.d_flat),
				id			= ii.run_id,
				tags 		= [ii.mode,],
				reinit 		= not (ii.wb.run is None)
			)

			print('c', ii.d)

	def pf_submit(ii):

		if ii.resource.submit:
			print('submitting')
			ii.resource.submit = False # docs:submit
			ii.debug_log([ii.d,], ['pf_submit.log',])

			run_or_sweep_d = ii.get_run_or_sweep_d()

			for i, run_d in enumerate(run_or_sweep_d):

				if ii.run_sweep or ii.wb.wb_sweep:
					group_exp = not i==0
				else:
					group_exp = ii.group_exp

				ii.setup_exp_dir(group_exp= group_exp, force_new_id= True)
				base_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, ignore=ii.ignore+['sweep','resource','dist_c','slurm_c'])
				run_d = base_d | run_d

				ii.debug_log([dict(os.environ.items()), run_d], ['env_submit.log', 'd_submit.log'])

				ii.resource.cluster_submit(run_d)

				print(ii.wb.run_url)
			
			sys.exit('Exiting from submit.')

	@staticmethod
	def pr(d: dict):
		""" pretty print and return dict """
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

	@property
	def sub_ins(ii):
		return dict()

	def init_sub_cls(ii,) -> dict:
		sub_cls = ins_to_dict(ii, sub_cls=True)
		for sub_k, sub_v in sub_cls.items():
			sub_ins = sub_v(parent=ii)
			setattr(ii, sub_k, sub_ins)
			if sub_k not in ii.sub_ins:
				ii.sub_ins[sub_k] = sub_ins

	def setup_exp_dir(ii, group_exp=False, force_new_id=False):

		if (not ii.exp_id) or force_new_id:
			ii.exp_id = gen_time_id(7)

			exp_name = ii.exp_name or 'junk'
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

	def to(ii, device, dtype):
		base_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, ignore=ii.ignore+['sweep'])
		d = {k:v for k,v in base_d.items() if isinstance(v, (np.ndarray, np.generic))}
		d = {k:torch.tensor(v, requires_grad=False).to(device=device, dtype=dtype) for k,v in d.items()}
		ii.update(d)
			
	def set_dtype(ii, dtype=torch.DoubleTensor):
		print('setting default dtype: ', dtype)
		torch.set_default_tensor_type(dtype) 
		ii.dtype = torch.randn((1,)).dtype

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
	
 
# slurm things
class niflheim_resource(PyfigBase.resource):
	_p: PyfigBase = None
 
	env: str     	= ''
	n_gpu: int 		= 1
	n_node: int		= 1

	architecture:   str 	= 'cuda'
	nifl_gpu_per_node: int  = property(lambda _: 10)
	device_log_path: str 	= '' 

	job_id: 		str  	= property(lambda _: os.environ.get('SLURM_JOBID', 'No SLURM_JOBID available.'))  # slurm only

	_pci_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
	pci_id:			str		= property(lambda _: ''.join(run_cmds(_._pci_id_cmd, silent=True)))

	n_device_env:	str		= 'CUDA_VISIBLE_DEVICES'
	# n_device:       int     = property(lambda _: sum(c.isdigit() for c in os.environ.get(_.n_device_env, '')))
	n_device:       int     = property(lambda _: len(os.environ.get(_.n_device_env, '').replace(',', '')))

	class slurm_c(Sub):
		export			= 'ALL'
		nodes           = '1' 			# (MIN-MAX) 
		cpus_per_gpu   = 8				# 1 task 1 gpu 8 cpus per task 
		partition       = 'sm3090'
		time            = '0-00:10:00'  # D-HH:MM:SS
		gres            = property(lambda _: 'gpu:RTX3090:' + (str(_._p.n_gpu) if int(_.nodes) == 1 else '10'))
		ntasks          = property(lambda _: _._p.n_gpu)
		job_name        = property(lambda _: _._p._p.exp_name)
		output          = property(lambda _: _._p._p.cluster_dir/'o-%j.out')
		error           = property(lambda _: _._p._p.cluster_dir/'e-%j.err')

	# mem_per_cpu     = 1024
	# mem				= 'MaxMemPerNode'
	# n_running_cmd:	str		= 'squeue -u amawi -t pending,running -h -r'
	# n_running:		int		= property(lambda _: len(run_cmds(_.n_running_cmd, silent=True).split('\n')))	
	# running_max: 	int     = 20

	def cluster_submit(ii, job: dict):
		
		if job['head']:
			print(ii._slurm)

		body = []
		body = f"""
		module purge
		module load foss
		source ~/.bashrc
  		conda activate {ii.env}
		export exp_id="{job["exp_id"]}"
		echo exp_id-${{exp_id}}
		"""
		extra = """
		module load CUDA/11.7.0'
		module load OpenMPI'
		export MKL_NUM_THREADS=1
		export NUMEXPR_NUM_THREADS=1
		export OMP_NUM_THREADS=8
		export OPENBLAS_NUM_THREADS=1
		"""
		debug_body = f""" \
  		export $SLURM_JOB_ID
		echo all_gpus-${{SLURM_JOB_GPUS}}', 'echo nodelist-${{SLURM_JOB_NODELIST}}', 'nvidia-smi']
		echo api-${{WANDB_API_KEY}}
		echo project-${{WANDB_PROJECT}}
		echo entity-${{WANDB_ENTITY}}
		echo ${{PWD}}
		echo ${{CWD}}
		echo ${{SLURM_EXPORT_ENV}}
		scontrol show config
		srun --mpi=list
		export WANDB_DIR="{ii._p.exp_dir}"
		printenv']
		curl -s --head --request GET https://wandb.ai/site
		ping api.wandb.ai
		"""
		if ii._p.wb.wb_sweep:
			body += [f'wandb controller {ii._p.wb.sweep_id}'] # {ii._p.wb.sweep_path_id}
			body += [f'wandb agent {ii._p.wb.sweep_id} 1> {ii.device_log_path(rank=0)} 2>&1 ']
			body += ['wait',]

		elif ii._p.distribute.dist_method == 'accelerate':
			print('\n accelerate distribution')
			# ! backslash must come between run.py and cmd
			cmd = dict_to_cmd(job, exclude_none=True)
			body += f'{ii._p.distribute._launch_cmd} {job["run_name"]} \ {cmd} 1> {ii.device_log_path(rank=0)} 2>&1 \n'  
   
		elif ii._p.distribute.dist_method == 'pyfig':
			print('\n pyfig distribution')
			for i in range(ii.n_gpu):
				job.update(dict(head= i==0))
				cmd = dict_to_cmd(job)
				cmd = f'python -u {job["run_name"]} {cmd}'
				body += f'\n{ii._p.distribute._launch_cmd} {cmd} 1> {ii.device_log_path(rank=i)} 2>&1 & \n'
			body += '\nwait \n'
		
		body += '\necho End \n'

		if ii._p.debug:
			print(body)

		ii._p.log([body,], ii._p.cluster_dir/'sbatch.log')
		job_id = ii._slurm.sbatch(body, verbose=True)
		print('slurm out: ', job_id)
  
	@property
	def _slurm(ii,) -> Slurm:
		if ii._p.debug:
			ii._p.pr(ii.slurm_c.d)
		return Slurm(**ii.slurm_c.d)

	def device_log_path(ii, rank=0):
		return ii._p.exp_dir/(str(rank)+"_device.log") # + ii._p.hostname.split('.')[0])

def setup_distribute(
		ii, 
		model: nn.Module, 
		opt, 
		tr_loader=None, 
		scheduler=None
) -> tuple:
		ii.dist = Accelerator()
		model, opt, tr_loader, scheduler = \
			ii.dist.prepare(model, opt, tr_loader, scheduler)
		
		return [model, opt] + (tr_loader or []) + (scheduler or [])

def backward(ii, loss: torch.Tensor, sync: dict):
	loss.backward()

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
			rb = np.load,
			wb = np.save,
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

class distribute_pyfig(Sub):
 
	def __init__(ii, parent=None):
		super().__init__(parent)
		global Accelerator
		global DataLoader

		from accelerate import Accelerator
		from torch.utils.data import DataLoader
 
	# class git(Sub):
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