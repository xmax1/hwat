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


from utils import dict_to_cmd, cmd_to_dict, dict_to_wandb
from utils import mkdir, iterate_n_dir, gen_time_id, add_to_Path, dump, load
from utils import get_cartesian_product, type_me, run_cmds, flat_any 
from dump.systems import systems

from torch import nn

this_dir = Path(__file__).parent
hostname = os.environ['HOSTNAME']

class Sub:
	_p = None
	ignore: list = ['_p',]
	
	def __init__(ii, parent=None):
		ii._p: PyfigBase
		ii._p: PyfigBase = parent
		ii.ignore += ii._p.ignore
  
	@property
	def d(ii):
		return inst_to_dict(ii, sub_cls=True, prop=True, attr=True, ignore=ii.ignore)

class Config:
    pass

class Static:
    pass

class Param(Sub, Static): 
    # docs:todo all wb sweep structure
	values: list = None
	domain: tuple = None
	dtype: type = None
	log: bool = False
	step_size: float|int = None
	sample: str = None # docs:Param:sample from ('uniform', )

	def __init__(ii, **kw) -> None: # docs:Param:init needed so can use kw arg to init
		for k,v in kw.items():
			setattr(ii, k, v)

	# @property
	# def d(ii):
	# 	return inst_to_dict(ii, attr=True, sub_cls=True, prop=True, ignore=ii.ignore)
		# return dict((k,v) for k,v in vars(ii).items() if (not isinstance(v, property)) and (not callable(v)))

class PyfigBase:

	user: 				str 	= None
 
	project:            str     = ''
	run_name:       	Path	= 'run.py'
	exp_name:       	str		= '' # default is demo
	exp_id: 			str		= ''
	group_exp: 			bool	= False

	multimode: 			str		= 'train:evaluate' # 'max_mem:profile:opt_hypam:train:evaluate'
	mode: 				str		= ''
	debug: 				bool    = False
	run_sweep:      	bool    = False
	
	seed:           	int   	= 808017424 # grr
	dtype:          	str   	= 'float32'

	n_step:         	int   	= 5000
	n_step_eval:        int   	= 1000
	n_pretrain_step:    int   	= 500

	log_metric_step:	int   	= 10
	log_state_step: 	int   	= 10
	
	class data(Sub):
		system: str = ''
		charge:     int         = 0
		spin:       int         = 0
		a:          np.ndarray  = np.array([[0.0, 0.0, 0.0],])
		a_z:        np.ndarray  = np.array([4.,])

		n_b:        int         = 256
		n_corr:     int         = 20
		n_equil:    int         = 10000
		acc_target: int         = 0.5

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
		opt_name: 		str		= 'RAdam'
		lr:  			float 	= 0.01
		betas:			list	= [0.9, 0.999]
		eps: 			float 	= 1e-4
		weight_decay: 	float 	= 0.0
		hessian_power: 	float 	= 1.
  
	class sweep(Sub):
		storage: 		Path = property(lambda _: 'sqlite:///' + str(_._p.exp_dir / 'hypam_opt.db'))
		sweep_name: 	str				= None
		sweep_method: 	str				= None # name of the alg 
		parameters: 	dict[Param]		= None
		n_trials: 		int				= 100
  
	class wb(Sub):
		run = None
		job_type:		str		= 'debug'		
		wb_mode: 		str		= 'disabled'
		wb_sweep: 		bool	= False
		_wb_agent: 		bool	= False
		sweep_id: 		str 	= ''
  
		entity:			str		= property(lambda _: _._p.project)
		program: 		Path	= property(lambda _: Path( _._p.project_dir, _._p.run_name))
		sweep_path_id:  str     = property(lambda _: f'{_.entity}/{_._p.project}/{_.sweep_id}')
		wb_type: 		str		= property(lambda _: _.wb_sweep*f'sweeps/{_.sweep_id}' or f'runs/{_._p.exp_id}') # _._p.group_exp*f'groups' or 'runs') # or _._p.run_sweep*f'groups/{_._p.exp_name}'
		run_url: 		str		= property(lambda _: f'https://wandb.ai/{_.entity}/{_._p.project}/{_.wb_type}')
  
	class distribute(Sub):
		dist: 			Any		= None
		dist_method: 	str		= 'accelerate'
		head:			bool	= True
		rank: 			bool 	= property(lambda _: os.environ.get('RANK', '0'))  #  or _.head no work bc same input to all script fix
		dist_mode: 		str		= 'pyfig'  # accelerate
		dist_id:		str		= ''
		sync_step:		int		= 5

		_gpu_id_cmd:		str	= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
		gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_._gpu_id_cmd, silent=True)).split('.')[0])
		dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])
		backward: 		Callable = None

	class resource(Sub):
		submit: 		bool	= False
		cluster_submit: Callable= None
		script:			Callable= None
		device_log_path:Callable= None

	home:				Path	= Path().home()
	project_dir:        Path    = property(lambda _: _.home / 'projects' / _.project)
	dump_dir:               Path    = property(lambda _: Path('dump'))
	tmp_dir:            Path	= property(lambda _: Path(_.dump_dir,'tmp'))
	exp_dir:        	Path	= property(lambda _: Path(_.dump_exp_dir, _.exp_name, _.exp_id))
	dump_exp_dir:        	Path	= property(lambda _: Path(_.dump_dir, 'exp'))
	cluster_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'cluster'))
	exchange_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'exchange'))
	profile_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'profile'))
	exp_data_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'exp_data'))
	log_dir: 			Path    = property(lambda _: _.cluster_dir)

	ignore_f = ['commit', 'pull', 'backward']
	ignore_cls = ['Static']
	ignore: list = ['ignore', 'd', 'cmd', 'sub_cls',] + ignore_f
	_sub_cls: list[Sub] = None

	def __init__(ii, 
		notebook:bool=False,  # removes sys_arg for notebooks
		sweep: dict={},  # special properties for config update so is separated
		c_init: dict|str|Path={},  # args specificall  
		post_init_arg: dict={},
		**other_arg):     

		ii._init_sub_cls()

		for k,v in (sweep or {}).items():
			setattr(ii.sweep, k, v)

		if not notebook:
			sys_arg = cmd_to_dict(sys.argv[1:], ii._get_d_with(attr=True, sub_cls=True, flat=True, ignore=ii.ignore+['sweep'])) 

		update = flat_any((c_init or {})) | flat_any((other_arg or {})) | (sys_arg or {})
		ii.update(update)

		ii.debug_log([sys_arg, dict(os.environ.items()), ii.d], ['log_sys_arg.log', 'log_env_run.log', 'log_d.log'])
		
	def __post_init__(ii, post_init_arg: dict=None, **other_arg):
		""" application specific initialisations """
		ii.update((post_init_arg or {}) | other_arg)


	def runfig(ii):

		ii.debug_log([ii.d,], ['runfig.log',])
  
		if not ii.resource.submit:
			if ii.distribute.head and int(ii.distribute.rank)==0:

				ii.setup_exp_dir(group_exp= ii.group_exp, force_new_id= False)

				if ii.wb.sweep_id:
					wandb.init()
					ii.debug_log([ii.d,], ['log_agent_arg.log',])
					# from run import run
					# wandb.agent(ii.wb.sweep_id, function=run, project=ii.project, entity=ii.wb.entity, count=1)
					# https://docs.wandb.ai/guides/sweeps/local-controller
					print('After Agent')
	
				else:
					ii.wb.run = wandb.init(
						project     = ii.project, 
						group		= ii.exp_name,
						dir         = ii.exp_data_dir,
						entity      = ii.wb.entity,  	
						mode        = ii.wb.wb_mode,
						config      = dict_to_wandb(ii.d),
						id			= ii.exp_id+str(len(list(ii.exp_data_dir.iterdir())))
					)
	
		elif ii.resource.submit:
			ii.resource.submit = False # docs:submit
   
			run_or_sweep_d = ii.get_run_or_sweep_d()

			for i, run_d in enumerate(run_or_sweep_d):

				if ii.run_sweep or ii.wb.wb_sweep:
					group_exp = not i==0
				else:
					group_exp = ii.group_exp

				ii.setup_exp_dir(group_exp= group_exp, force_new_id= True)
				base_d = inst_to_dict(ii, attr=True, sub_cls=True, flat=True, ignore=ii.ignore+['sweep', 'resource',], debug=ii.debug)
				run_d = base_d | run_d

				ii.debug_log([dict(os.environ.items()), run_d], ['env_submit.log', 'd_submit.log'])

				ii.resource.cluster_submit(run_d)

				print(ii.wb.run_url)
			
			ii.pr(ii._debug_paths)
			sys.exit('Exiting from submit.')

	@staticmethod
	def pr(d: dict):
		""" pretty print and return dict """
		pprint.pprint(d)
		return d

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
		return inst_to_dict(ii, sub_cls=True, prop=True, attr=True, ignore=ii.ignore)

	def _get_d_with(ii, sub_cls=False, prop=False, attr=False, flat=False, ignore=None):
		return inst_to_dict(ii, sub_cls=sub_cls, prop=prop, attr=attr, flat=flat, ignore=ignore)

	def _init_sub_cls(ii,) -> dict:
		_sub_cls = dict(
      		((k,v) for k,v in vars(ii.__class__).items() 
                if isinstance(v, type) and issubclass(v, Sub))
    	)
		for sub_name, sub in _sub_cls.items(): # docs:sub_classes-init
			setattr(ii, sub_name, sub(parent=ii))
	
	@property
	def sub_cls(ii):
		return dict((k,v) for k,v in vars(ii).items() if isinstance(v, Sub))

	def setup_exp_dir(ii, group_exp=False, force_new_id=False):

		if ii.debug:
			print('debug:setup_exp_dir:', ii.exp_id, ii.distribute.head, ii.group_exp, force_new_id)

		if (not ii.exp_id) or force_new_id:
			ii.exp_id = gen_time_id(7)

			exp_name = ii.exp_name or 'junk'
			sweep_dir = 'sweep'*ii.run_sweep
			exp_group_dir = Path(ii.dump_exp_dir, sweep_dir, exp_name)
			exp_group_dir = iterate_n_dir(exp_group_dir, group_exp=group_exp) # does not append -{i} if group allowed
			ii.exp_name = exp_group_dir.name

			print('exp_dir: ', ii.exp_dir)  # is property
   
		[mkdir(p) for _, p in ii._paths.items()]
	
	def get_run_or_sweep_d(ii,):
		
		if not (ii.run_sweep or ii.wb.wb_sweep):
			""" single run
			takes configuration from base in submit loop
			"""
			return [dict(),] 
	
		if ii.wb.wb_sweep:
			param = ii.sweep.parameters
			sweep_keys = list(param.keys())

			n_sweep = 0
			for k, k_d in param.items():
				v = k_d.get('values', [])
				n_sweep += len(v)
	
			# n_sweep = len(get_cartesian_product(*(v for v in param))
			base_c = inst_to_dict(ii, sub_cls=True, attr=True, flat=True, ignore=ii.ignore+['sweep', 'head', 'exp_id'] + sweep_keys)
			base_cmd = dict_to_cmd(base_c, sep='=')
			base_sc = dict((k, dict(value=v)) for k,v in base_c.items())
   
			if ii.wb.wb_sweep:
				sweep_c = dict(
					command 	= ['python', '-u', '${program}', f'{base_c}', '${args}', '--exp_id=${exp_id}', ],
					program 	= str(Path(ii.run_name).absolute()),
					method  	= ii.sweep.method,
					parameters  = base_sc|param,
					controller  = dict(type='local'),
				)

				os.environ['WANDB_PROJECT'] = 'hwat'
				os.environ['WANDB_ENTITY'] = 'hwat'

				pprint.pprint(sweep_c)

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

		ii.pr(ii.sweep.d)
		d = ii.sweep.d
		sweep_keys = list(d['parameters'].keys())
		sweep_vals = [v['values'] for v in d['parameters'].values()]
		sweep_vals = get_cartesian_product(*sweep_vals)
		print(f'### sweep over {sweep_keys} ({len(sweep_vals)} total) ###')
		return [{k:v for k,v in zip(sweep_keys, v_set)} for v_set in sweep_vals]

	def debug_log(ii, d_all:list, name_all: list):
		for d, name in zip(d_all, name_all):
			if Path(ii.exp_dir).is_dir():
				ii.log(d, path=ii.log_dir/name)
			ii.log(d, path=ii.tmp_dir/name)
			
	def partial(ii, f:Callable, args=None, **kw):
		d = flat_any(args if args else ii.d)
		d_k = inspect.signature(f.__init__).parameters.keys()
		d = {k:v for k,v in d.items() if k in d_k} | kw
		return f(**d)

	def update(ii, arg: dict):
		arg = flat_any(arg)
		for k,v in copy(arg).items():
			for inst in [ii,] + list(ii.sub_cls.values()):
				if hasattr(inst, k):
					v_ref = getattr(inst, k)
					v = type_me(v, v_ref)
					setattr(inst, k, copy(arg.pop(k)))
					print(f'update {k}: {v_ref} --> {v}')
		
		print('not updated:')
		[print(f'! ', (k, v, type(v))) for k,v in arg]

	# def __setattr__(ii, k: str, v: Any):
	# 	for inst in [ii,]
	# 		ref = ii.__dict__[k]
	# 		ii.__dict__[k] = v

	@staticmethod
	def log(info: dict|str, path: Path):
		mkdir(path)
		info = pprint.pformat(info)
		with open(path, 'w') as f:
			f.writelines(info)

	def to(ii, framework='torch'):
		if framework=='torch':
			base_d = ii._get_d_with(attr=True, sub_cls=True, flat=True, ignore=ii.ignore+['sweep'])
			d = {k:v for k,v in base_d.items() if isinstance(v, (np.ndarray, np.generic, list))}
			d = {k:torch.tensor(v, requires_grad=False) for k,v in d.items() if not isinstance(v[0], str)}
		ii.update(d)
			
	def set_dtype(ii):
		dtype = torch.randn((1,)).dtype
		ii.update({k:v.to(dtype) for k,v in flat_any(ii.d).items() if isinstance(v, torch.Tensor)})

	def set_dist(ii, dist: Any=None):
		print('setting distribution: ', dist)
		print('config says ', ii.distribute.dist_method)
		ii.distribute.dist = dist
		return dist

	def set_device(ii, device=None):
		device = device or ii.distribute.dist.device
		ii.update({k:v.to(device) for k,v in flat_any(ii.d).items() if isinstance(v, torch.Tensor)})
		return device
  




class distribute_accelerate(Sub):
	
	def __init__(ii, parent=None):
		super().__init__(parent)
		global Accelerator
		global DataLoader

		from accelerate import Accelerator
		from torch.utils.data import DataLoader

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
		ii.dist.backward(loss)
  
class distribute_pyfig(Sub):
	_p: PyfigBase
 
	def __init__(ii, parent=None):
		super().__init__(parent)
		global Accelerator
		global DataLoader

		from accelerate import Accelerator
		from torch.utils.data import DataLoader

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
  
	def sync(ii, step: int, v_tr: dict):
		v_path = (ii._p.exchange_dir / f'{step}_{ii._p.distribute.dist_id}').with_suffix('.pk')
		v_mean_path = add_to_Path(v_path, '-mean')
		
		try:
			gc.disable()

			v_ref_leaves, treespec = optree.tree_flatten(v_tr)
			v_sync_save = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in v_ref_leaves]
			dump(v_path, v_sync_save)

		except Exception as e:
			print(e)
		finally:
			gc.enable()
		
		if ii._p.distribute.head:

			n_ready = 0
			while n_ready < ii.resource.n_gpu:
				k_path_all = list(ii.exchange_dir.glob(f'{step}_*'))
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
			v_sync = v_tr
			print(e)
		finally: # ALWAYS EXECUTED
			v_mean_path.unlink()
			gc.enable()
		return v_sync


### slurm things

class niflheim_resource(Sub):
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
	n_device:       int     = property(lambda _: sum(c.isdigit() for c in os.environ.get(_.n_device_env, 'ZERO')))

	class c(Config):
		export			= 'ALL'
		nodes           = '1' 			# (MIN-MAX) 
		# mem_per_cpu     = 1024
		# mem				= 'MaxMemPerNode'
		cpus_per_gpu   = 8				# 1 task 1 gpu 8 cpus per task 
		partition       = 'sm3090'
		time            = '0-00:10:00'  # D-HH:MM:SS
		gres            = property(lambda _: 'gpu:RTX3090:' + (str(_.n_gpu) if int(_.nodes) == 1 else '10'))
		ntasks          = property(lambda _: _.n_gpu)
		job_name        = property(lambda _: _._p.exp_name)
		output          = property(lambda _: _._p.cluster_dir/'o-%j.out')
		error           = property(lambda _: _._p.cluster_dir/'e-%j.err')

	# n_running_cmd:	str		= 'squeue -u amawi -t pending,running -h -r'
	# n_running:		int		= property(lambda _: len(run_cmds(_.n_running_cmd, silent=True).split('\n')))	
	# running_max: 	int     = 20

	def cluster_submit(ii, job: dict):
		
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
	  
		if job['head']:
			print(ii.script)
		body = []
		body += [
			'module purge', 
			'module load foss', 
   			# 'module load CUDA/11.7.0',
   			# 'module load OpenMPI',
		]
		# body += [f'source ~/.bashrc', ]
		# body += [f'conda activate {ii.env}',]
		# body += [
		# 'export $SLURM_JOB_ID',
		# 'export MKL_NUM_THREADS=1',
		# 'export NUMEXPR_NUM_THREADS=1',
		# 'export OMP_NUM_THREADS=8',
		# 'export OPENBLAS_NUM_THREADS=1',
		# ]
		body += ['echo all_gpus-${SLURM_JOB_GPUS}', 'echo nodelist-${SLURM_JOB_NODELIST}', 'nvidia-smi']
		
		port = np.random.randint(29500, 64000)
		# body += ['export ']
		body += ['echo api-${WANDB_API_KEY}']
		body += ['echo project-${WANDB_PROJECT}']
		body += ['echo entity-${WANDB_ENTITY}']
		# body += ['curl -s --head  --request GET https://wandb.ai/site']
		body += [f'export exp_id="{job["exp_id"]}"', 'echo exp_id-${exp_id}']
		body += ['echo ${PWD}']
		body += ['echo ${CWD}']
		body += ['echo ${SLURM_EXPORT_ENV}']
		body += ['scontrol show config']
		body += ['srun --mpi=list']
		body += [f'export WANDB_DIR="{ii._p.exp_dir}"']
		body += ['printenv']
		# body += [f'ping api.wandb.ai']

		if ii._p.wb.wb_sweep:
			print('\n wb sweep')
				# server local https://github.com/wandb/wandb/issues/4586
			# launch_cmd = 'srun --gpus=1 --ntasks=1 --exclusive --label  '
			launch_cmd = ''
			body += [f'wandb controller {ii._p.wb.sweep_id}'] # {ii._p.wb.sweep_path_id}
			body += [f'wandb agent {ii._p.wb.sweep_id} 1> {ii.device_log_path(rank=0)} 2>&1 ']
			# body += [f'{launch_cmd} python -u {ii._p.run_name} {dict_to_cmd(job)} 1> {ii.device_log_path(rank=0)} 2>&1 & ']
				# srun --nodes=1 --mpi=cray_shasta --gpus=1 --cpus-per-task=8 --ntasks=1 -vvvvv --label --exclusive bash -c &
			body += ['wait',]
   
		elif ii._p.distribute.dist_method == 'accelerate':
			print('\n accelerate distribution')
			# c_dist = lo_ve(path='c_dist.yaml')
			c_dist = 'c_dist.yaml'
			launch_cmd = f'accelerate launch --config_file {c_dist} --main_process_port {port}'
			cmd = dict_to_cmd(job)
			body += [f'{launch_cmd} {job["run_name"]} \ {cmd} 1> {ii.device_log_path(rank=0)} 2>&1 '] # \  ']  # \n 
   
			# backslash must come between run.py and cmd
		elif ii._p.distribute.dist_method == 'pyfig':
			print('\n pyfig distribution')
			launch_cmd = 'srun --gpus=1 --cpus-per-task=4 --ntasks=1 --exclusive --label  '
			for i in range(ii.n_gpu):
				job.update(dict(head= i==0))
				cmd = dict_to_cmd(job)
				cmd = f'python -u {job["run_name"]} {cmd}'
				body += [f'{launch_cmd} {cmd} 1> {ii.device_log_path(rank=i)} 2>&1 & ']
		
			body += ['wait',]
   
		body += ['echo End']
		body = '\n'.join(body)
		if ii._p.debug:
			print(body)
		slurm = ii.script
		ii._p.log([body,], ii._p.cluster_dir/'sbatch.log')
		job_id = slurm.sbatch(body, verbose=True)
		print('slurm out: ', job_id)
  
	@property
	def script(ii,):
		return Slurm(
			export			= ii.export,
			partition       = ii.partition,
			nodes           = ii.nodes,
			cpus_per_gpu   	= ii.cpus_per_gpu,
			time            = ii.time         ,
			gres            = ii.gres         ,
			ntasks          = ii.ntasks       ,
			job_name        = ii.job_name     ,
			output          = ii.output       ,
			error           = ii.error        ,
		)

	def device_log_path(ii, rank=0):
		return ii._p.cluster_dir/(str(rank)+"_device.log") # + ii._p.hostname.split('.')[0])



""" slurm docs
sinfo -p cluster
groups
sbalance
sreport -t hours cluster AccountUtilization account=project_465000153
sbatch - submit a batch script
salloc - allocate compute resources
srun - allocate compute resources and launch job-steps
squeue - check the status of running and/or pending jobs
scancel - delete jobs from the queue
sinfo - view intormation abount cluster nodes and partitions
scontrol - show detailed information on active and/or recently completed jobs, nodes and partitions
sacct - provide the accounting information on running and completed jobs
slurmtop - text-based view of cluster nodes' free and in-use resources and status of jobs

Based on available resources and in keeping with maintaining a fair balance between all users, we may sometimes be able to accommodate special needs for a limited time. In that case, please submit a short explanation to cluster-help@luis.uni-hannover.de.

To list job limits relevant for you, use the sacctmgr command:

sacctmgr -s show user
sacctmgr -s show user adwilson (works on lumi)
sacctmgr -s show user format=user,account,maxjobs,maxsubmit,maxwall,qos
sacctmgr -s show user zailacka
Up-to-date information on ALL available nodes:

 sinfo -Nl
 scontrol show nodes
Information on partitons and their configuration:

 sinfo -s
 scontrol show partitions

"""

# utils things we might move 

import pickle as pk
import yaml
import json
from functools import partial

def lo_ve(path:Path, data=None):
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

	with open(path, mode) as f:
		data = interface(data, f) if not data is None else interface(f)
	return data


def inst_to_dict(
	inst: PyfigBase, 
	attr=False,
	sub_cls=False, 
	prop=False,
	ignore:list=None,
	flat:bool=False,
	debug:bool=False
) -> dict:
		ignore = ignore or []

		inst_keys = [k for k in dir(inst) if not k.startswith('_') and not k in ignore] # docs:python:cls-inst:q? could dirs --> vars? 
		if debug:
			print('inst-to-dict: inst_keys: ', inst_keys)


		d_ins = {k:v for k,v in vars(inst).items() if not k.startswith('_') and not k in ignore}
		d_ins = {k:getattr(inst, k) for k in inst_keys}
		d_cls = {k:getattr(inst.__class__, k) for k in inst_keys}
		
		d_callable = {k:v for k,v in d_ins.items() if callable(v) and not isinstance(v, Sub)}
		d_prop = {k:v for k,v in d_ins.items() if isinstance(d_cls[k], property)}
		property_has_setter = lambda v: getattr(v, 'fset') is not None
		d_prop_w_setter = {k:v for k,v in d_prop.items() if property_has_setter(d_cls[k])}
		
		d_sub = {}
		for k,v in d_ins.items():
			if isinstance(v, Sub):
				d_sub[k] = inst_to_dict(v, attr=attr, sub_cls=sub_cls, prop=prop, ignore=ignore)
	
		d_attr = {k:v for k,v in d_ins.items() if not k in (d_callable | d_prop | d_sub).keys()} | d_prop_w_setter
		
		d = dict()
		[d.setdefault(k, v) for k,v in d_sub.items() if sub_cls]
		[d.setdefault(k, v) for k,v in d_attr.items() if attr]
		[d.setdefault(k, v) for k,v in d_prop.items() if prop]
		if debug:
			print('inst-to-dict: d_sub: ', d_sub)
		return flat_any(d) if flat else d # docs:python:flat:q? parameter to flat dict in levels?

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
 
	# class git(Sub):
	# 	branch:     str     = 'main'
	# 	remote:     str     = 'origin' 

	# 	_commit_id_cmd:	str 	= 'git log --pretty=format:%h -n 1'
	# 	commit_id:   	list	= property(lambda _: run_cmds(_._commit_id_cmd, cwd=_._p.project_dir, silent=True))
		# commit_cmd:	str     = 'git commit -a -m "run"' # !NB no spaces in msg 
		# commit: 		list	= property(lambda _: run_cmds(_.commit_cmd, cwd=_.project_dir)) 
		# pull_cmd:		str 	= ['git fetch --all', 'git reset --hard origin/main']
		# pull:   		list	= property(lambda _: run_cmds(_.pull_cmd, cwd=_.project_dir))
  
  
"""