from pathlib import Path
from itertools import islice
from time import sleep
import optree
import os
import gc
from simple_slurm import Slurm
import sys
from typing import Callable
import wandb
import pprint
import inspect
import numpy as np
from copy import copy
import torch 

from utils import dict_to_cmd, cmd_to_dict, dict_to_wandb
from utils import torchify_tree, numpify_tree
from utils import mkdir, iterate_n_dir, gen_time_id, add_to_Path, dump, load
from utils import get_cartesian_product, type_me, run_cmds, flat_any 
from dump.systems import systems

"""
todo
- single function for cmd calls

docs:sub_classes-init
Steps
0- subclasses inherit from personal bases
1- initialised subclasses, because they don't have properties otherwise

## docs:submit
- must change submit to False to prevent auto resubmitting

"""
this_dir = Path(__file__).parent
hostname = os.environ['HOSTNAME']

class Sub:
	_p = None
	ignore: list = ['ignore', 'd', '_p',]
	
	def __init__(ii, parent=None):
		ii._p = parent
  
	@property
	def d(ii,):
		return inst_to_dict(ii, sub_cls=True, flat=True, prop=True, ignore=ii.ignore)

class PyfigBase:

	user: 				str 	= None
 
	project:            str     = ''
	run_name:       	Path	= 'run.py'
	exp_name:       	str		= 'demo'
	exp_dir:        	Path	= ''
	exp_id: 			str		= ''
	group_exp: 			bool	= False
	
	seed:           	int   	= 808017424 # grr
	dtype:          	str   	= 'float32'
	n_step:         	int   	= 10000
	log_metric_step:	int   	= 10
	log_state_step: 	int   	= 10
	
	class data(Sub):

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
		lr: 			float	= 0.001
		name: 			str		= 'Adam'
  
	class sweep(Sub):
		run_sweep:      bool    = False	
		method: 		str		= 'grid'
		parameters: 	dict 	= 	dict(
			n_b  = dict(values=[16, 32, 64]),
		)
  
	class wb(Sub):
		run = None
		job_type:		str		= 'debug'		
		wb_mode: 		str		= 'disabled'
		wb_sweep: 		bool	= False
  
		entity:			str		= property(lambda _: _._p.project)
		program: 		Path	= property(lambda _: Path( _._p.project_dir, _._p.run_name))
		sweep_path_id:  str     = property(lambda _: f'{_.entity}/{_._p.project}/{_._p.exp_name}')
		wb_type: 		str		= property(lambda _: _.wb_sweep*'sweeps' or 'groups') # _._p.group_exp*f'groups' or 'runs')
		run_url: 		str		= property(lambda _: f'https://wandb.ai/{_.entity}/{_._p.project}/{_.wb_type}/{_._p.exp_name}/')
  
	class distribute(Sub):
		head:			bool	= True 
		dist_mode: 		str		= 'pyfig'  # accelerate
		dist_id:		str		= ''
		sync_step:		int		= 5

		_gpu_id_cmd:		str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
		gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_._gpu_id_cmd, silent=True)).split('.')[0])
		dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])

	class resource(Sub):
		submit: 		bool	= False
		cluster_submit: Callable= None
		script:			Callable= None

	class git(Sub):
		branch:     str     = 'main'
		remote:     str     = 'origin' 

		_commit_id_cmd:	str 	= 'git log --pretty=format:%h -n 1'
		commit_id:   	list	= property(lambda _: run_cmds(_._commit_id_cmd, cwd=_._p.project_dir, silent=True))
		# commit_cmd:	str     = 'git commit -a -m "run"' # !NB no spaces in msg 
		# commit: 		list	= property(lambda _: run_cmds(_.commit_cmd, cwd=_.project_dir)) 
		# pull_cmd:		str 	= ['git fetch --all', 'git reset --hard origin/main']
		# pull:   		list	= property(lambda _: run_cmds(_.pull_cmd, cwd=_.project_dir))
  
	home:				Path	= Path().home()
	dump:               Path    = property(lambda _: Path('dump'))
	dump_exp_dir: 		Path 	= property(lambda _: _.dump/'exp')
	tmp_dir:            Path	= property(lambda _: _.dump/'tmp')
	project_dir:        Path    = property(lambda _: _.home / 'projects' / _.project)
	cluster_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'cluster'))
	exchange_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'exchange'))
	profile_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'profile'))
	log_dir: 			Path    = property(lambda _: _.cluster_dir)


	debug: bool    = False
	env_log_path = 'dump/tmp/env.log'
	d_log_path = 'dump/tmp/d.log'

	ignore: list = [
		'ignore', 
		'd', 'cmd', 'sub_cls', 
		'sweep',
		'commit', 'pull',
	]

	def __init__(ii, notebook:bool=False, sweep: dict=None, **init_arg):     

		for sub_name, sub in ii.sub_cls.items(): # docs:sub_classes-init
			setattr(ii, sub_name, sub(parent=ii))

		c_init = flat_any(ii.d)
		sys_arg = sys.argv[1:]
		sys_arg = cmd_to_dict(sys_arg, c_init) if not notebook else {}  
		init_arg = flat_any(init_arg) | (sweep or {})
   
		ii.update_configuration(init_arg | sys_arg)

		ii.setup_exp_dir(group_exp= ii.group_exp, force_new_id= False)

		ii.debug_log([dict(os.environ.items()), ii.d,], ['env.log', 'd.log'])
  
	def runfig(ii):

		if (not ii.resource.submit) and ii.distribute.head:
			ii.wb.run = wandb.init(
				project     = ii.project, 
				group		= ii.exp_name,
				id          = ii.exp_id,
				dir         = ii.exp_dir,
				entity      = ii.wb.entity,  	
				mode        = ii.wb.wb_mode,
				config      = dict_to_wandb(ii.d),
			)
		
		if ii.resource.submit:
			ii.resource.submit = False # docs:submit

			run_or_sweep_d = ii.get_run_or_sweep_d()
			for i, run_d in enumerate(run_or_sweep_d):
				ii.distribute.head = i == 0
				group_exp = ii.group_exp or (ii.distribute.head and ii.sweep.run_sweep)
				
				ii.setup_exp_dir(group_exp= group_exp, force_new_id= True)
				if ii.distribute.head:
					ii.debug_log([dict(os.environ.items()), run_d], ['env_submit.log', 'd_submit.log'])
     
				base_d = inst_to_dict(ii, attr=True, sub_cls=True, flat=True, ignore=ii.ignore, debug=ii.debug)
				run_d = base_d | run_d

				ii.resource.cluster_submit(run_d)

			sys.exit(ii.wb.run_url + ii.exp_id*(not i))
	
	@staticmethod
	def pr(d: dict):
		pprint.pprint(d)

	@property
	def cmd(ii):
		return dict_to_cmd(ii.d)

	@property
	def d(ii):
		return inst_to_dict(ii, sub_cls=True, prop=True, attr=True, ignore=ii.ignore)

	@property
	def sub_cls(ii) -> dict:
		inst_keys = [k for k in dir(ii) if not k.startswith('_') and not k in ii.ignore]
		ii.log(inst_keys, path=ii.tmp_dir/'inst_keys.log')
		d_init = {k:getattr(ii, k) for k in inst_keys}
		return {k:v for k,v in d_init.items() if isinstance(v, type) or isinstance(v, Sub)}

	def setup_exp_dir(ii, group_exp=False, force_new_id=False):

		if ii.debug:
			print('debug:setup_exp_dir:', ii.exp_dir, ii.distribute.head, ii.group_exp, force_new_id)

		if Path(ii.exp_dir).exists() and not force_new_id:
			return None
		exp_name = ii.exp_name or 'junk'
		exp_group_dir = Path(ii.dump_exp_dir, 'sweep'*ii.sweep.run_sweep, exp_name)
		exp_group_dir = iterate_n_dir(exp_group_dir, group_exp=group_exp)
		ii.exp_name = exp_group_dir.name
		ii.exp_id = (not force_new_id)*ii.exp_id or (ii.exp_id+'-'+gen_time_id(7) if ii.exp_id else gen_time_id(7))
		ii.exp_dir = exp_group_dir/ii.exp_id
		[mkdir(ii.exp_dir/_dir) for _dir in ['cluster', 'exchange', 'wandb', 'profile']]
	
	def get_run_or_sweep_d(ii,):
		
		if not ii.sweep.run_sweep:
			""" single run
			takes configuration from base in submit loop
			"""
			return [dict(),] 

		d = ii.sweep.d
		sweep_keys = list(d['parameters'].keys())
		sweep_vals = [v['values'] for v in d['parameters'].values()]
		sweep_vals = get_cartesian_product(*sweep_vals)
		print(f'### sweep over {sweep_keys} ({len(sweep_vals)} total) ###')
		return [{k:v for k,v in zip(sweep_keys, v_set)} for v_set in sweep_vals]

	def debug_log(ii, d_all:list, name_all: list):
		for d, name in zip(d_all, name_all):
			ii.log(d, path=ii.log_dir/name)
			ii.log(d, path=ii.tmp_dir/name)
			
	def partial(ii, f:Callable, args=None, **kw):
		d = flat_any(args if args else ii.d)
		d_k = inspect.signature(f.__init__).parameters.keys()
		d = {k:v for k,v in d.items() if k in d_k} | kw
		return f(**d)

	def update_configuration(ii, merge: dict, sweep: dict=None):
		merge = flat_any(merge) | dict(sweep=sweep)
		for k,v in merge.items():
			for inst in [ii,] + list(ii.sub_cls.values()):
				ref = inst_to_dict(inst, attr=True, ignore=ii.ignore)
				v_ref = ref.get(k, None)
				if not v_ref is None:
					v = type_me(v, v_ref)
					setattr(inst, k, copy(v))
					print(f'update {k}: {v_ref} --> {v}')
	
	@staticmethod
	def log(info: dict|str, path: Path):
		mkdir(path)
		info = pprint.pformat(info)
		with open(path, 'w') as f:
			f.writelines(info)

	def to(ii, to, device, dtype):
		if to == 'torch':
			import torch
			base_d = inst_to_dict(ii, attr=True, sub_cls=True, flat=True, ignore=ii.ignore, debug=ii.debug)
			d = {k:v for k,v in base_d.items() if isinstance(v, (np.ndarray, np.generic, list))}
			d = {k:torch.tensor(v, dtype=dtype, device=device, requires_grad=False) for k,v in d.items() if not isinstance(v[0], str)}
		ii.debug_log([d,], ['to_torch.log',])
		ii.update_configuration(d)
  
	def sync(ii, step: int, v_tr: dict):
		v_path = (ii.exchange_dir / f'{step}_{ii.distribute.dist_id}').with_suffix('.pk')
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
		
		if ii.distribute.head:

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
			v_sync_leaves = [torch.tensor(data=v, device=ref.device, dtype=ref.dtype) 
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

	architecture:   str 	= 'cuda'
	nifl_gpu_per_node: int  = property(lambda _: 10)

	job_id: 		str  	= property(lambda _: os.environ.get('SLURM_JOBID', 'No SLURM_JOBID available.'))  # slurm only

	_pci_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
	pci_id:			str		= property(lambda _: ''.join(run_cmds(_._pci_id_cmd, silent=True)))

	n_device_env:	str		= 'CUDA_VISIBLE_DEVICES'
	n_device:       int     = property(lambda _: sum(c.isdigit() for c in os.environ.get(_.n_device_env, 'ZERO')))

	### Slurm Configuration ###
	export			= 'ALL'
	nodes           = '1' 			# (MIN-MAX) 
	# mem_per_cpu     = 1024
	# mem				= 'MaxMemPerNode'
	cpus_per_task   = 8				# 1 task 1 gpu 8 cpus per task 
	partition       = 'sm3090'
	time            = '0-00:20:00'  # D-HH:MM:SS

	gres            = property(lambda _: 'gpu:RTX3090:' + (str(_.n_gpu) if int(_.nodes) == 1 else '10'))
	ntasks          = property(lambda _: _.n_gpu)
	job_name        = property(lambda _: _._p.exp_name)
	output          = property(lambda _: _._p.cluster_dir/'o-%j.out')
	error           = property(lambda _: _._p.cluster_dir/'e-%j.err')
 
	slurm: Slurm = None
	# n_running_cmd:	str		= 'squeue -u amawi -t pending,running -h -r'
	# n_running:		int		= property(lambda _: len(run_cmds(_.n_running_cmd, silent=True).split('\n')))	
	# running_max: 	int     = 20

	def script(ii, job: dict):
		ii.slurm = Slurm(
			export			= ii.export,
			nodes           = ii.nodes        ,
			# mem_per_cpu     = ii.mem_per_cpu  ,
			# mem     		= ii.mem  ,
			cpus_per_task   = ii.cpus_per_task,
			partition       = ii.partition,
			time            = ii.time         ,
			gres            = ii.gres         ,
			ntasks          = ii.ntasks       ,
			job_name        = ii.job_name     ,
			output          = ii.output       ,
			error           = ii.error        ,
		)
		if job['head']:
			print(ii.slurm)

		mod = ['module purge', 'module load foss', 'module load CUDA/11.7.0']
		env = ['source ~/.bashrc', f'conda activate {ii.env}',]
		export = ['export $SLURM_JOB_ID',]
		debug = ['echo all_gpus:$SLURM_JOB_GPUS', 'echo nodelist:$SLURM_JOB_NODELIST', 'nvidia-smi']
		srun_cmd = 'srun --gpus=1 --cpus-per-task=4 --ntasks=1 --exclusive --label '
		body = mod + env + debug
 
		for i in range(ii.n_gpu):
			
			device_log_path = ii._p.cluster_dir/(str(i)+"_device.log") # + ii._p.hostname.split('.')[0])

			job.update(dict(head= i==0))
   
			cmd = dict_to_cmd(job)
   
			cmd = f'python -u {job["run_name"]} {cmd}'
				
			body += [f'{srun_cmd} {cmd} 1> {device_log_path} 2>&1 & ']
   
		body += ['wait',]
		return '\n'.join(body)

	def cluster_submit(ii, job):
		sbatch = ii.script(job)
		ii._p.log([sbatch,], ii._p.tmp_dir/'sbatch.log')
		ii.slurm.sbatch(sbatch)

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

def inst_to_dict(
	inst, 
	attr=False,
	sub_cls=False, 
	prop=False,
	ignore:list=[],
	flat:bool=False,
	debug:bool=False
) -> dict:
		inst_keys = [k for k in dir(inst) if not k.startswith('_') and not k in ignore]
		
		d_cls = {k:getattr(inst.__class__, k) for k in inst_keys}
		d_ins = {k:getattr(inst, k) for k in inst_keys}
		
		d_callable = {k:v for k,v in d_ins.items() if callable(v) and not isinstance(v, Sub)}
		d_prop = {k:v for k,v in d_ins.items() if isinstance(d_cls[k], property)}
		
		d_sub = {}
		for k,v in d_ins.items():
			if isinstance(v, Sub):
				d_sub[k] = inst_to_dict(v, attr=attr, sub_cls=sub_cls, prop=prop, ignore=ignore)
		
		d_attr = {k:v for k,v in d_ins.items() if not k in (d_callable | d_prop | d_sub).keys()}
		
		d = dict()
		[d.setdefault(k, v) for k,v in d_sub.items() if sub_cls]
		[d.setdefault(k, v) for k,v in d_attr.items() if attr]
		[d.setdefault(k, v) for k,v in d_prop.items() if prop]
		if debug:
			print('inst-to-dict: inst_keys: ', inst_keys)
			print('inst-to-dict: d_sub: ', d_sub)
		return flat_any(d) if flat else d

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