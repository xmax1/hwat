import sys
import gc
from pathlib import Path
from typing import Union, Callable, Iterable, Any


from copy import deepcopy, copy
from itertools import product, islice
from functools import partial
import random
import re
from ast import literal_eval
import inspect
import os
import pprint
from time import time, sleep
import subprocess

import paramiko
import wandb
import optree
import torch
import numpy as np

dl_arr = torch.Tensor

this_dir = Path(__file__).parent
hostname = os.environ['HOSTNAME']

### exit handling ###
# NB: !Important! Distribution if 1 gpu fails the others continue without this
import atexit

def exit_handler():
	try:
		run_cmds(f'scancel {os.environ["SLURM_JOBID"]}')
	except Exception as e:
		print('Exiting boop beep bap.')
	
atexit.register(exit_handler)

### data save and load
import pickle as pk
import yaml
import json

file_interface_all = dict(
	pk = dict(
		rb = partial(pk.load, protocol=pk.HIGHEST_PROTOCOL),
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
	# deepdish = dict(
	# 	r = dd.io.load,
	# 	w = dd.io.save,
	# ),
)      

def data_lo_ve(path:Path, data=None):
	if not path.suffix:
		path = path.with_suffix('.pk')
	file_type = path.suffix[1:]
	mode = 'r' if data is None else 'w'
	mode += 'b' if file_type in ['pk',] else ''
	interface = file_interface_all[file_type][mode]
	with open(path, mode) as f:
		data = interface(data, f) if not data is None else interface(f)
	return data

def load(path):
	with open(path, 'rb') as f:
		data = pk.load(f)
	return data

def dump(path, data):
	with open(path, 'wb') as f:
		pk.dump(data, f, protocol=pk.HIGHEST_PROTOCOL)
	return

### metrics ###

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

### debug things

def debug_mode(on=False):
	if on:
		os.environ['debug'] = 'debug'
	else:
		os.environ['debug'] = ''

def debug_pr(d:dict):
	if os.environ.get('debug') == 'debug':
		for k,v in d.items():
			typ = type(v) 
			has_shape = hasattr(v, 'shape')
			shape = v.shape if has_shape else None
			dtype = v.dtype if hasattr(v, 'dtype') else None
			try:
				mean = torch.mean(v) if has_shape else v
				std = torch.std(v) if has_shape else None
			except:
				mean = v
				std = None
			print(k, f'\t mean={mean} \t std={std} \t shape={shape} \t dtype={dtype}') # \t type={typ}

### count things

def get_cartesian_product(*args):
	""" Cartesian product is the ordered set of all combinations of n sets """
	return list(product(*args))

def zip_in_n_chunks(arg: Iterable[Any], n: int) -> zip:   
	return zip(*([iter(arg)]*n))

def gen_alphanum(n: int = 7, test=False):
	from string import ascii_lowercase, ascii_uppercase
	random.seed(test if test else None)
	numbers = ''.join([str(i) for i in range(10)])
	characters = ascii_uppercase + ascii_lowercase + numbers
	name = ''.join([random.choice(characters) for _ in range(n)])
	return name

def gen_time_id(n=7):
	return str(round(time() * 1000))[-n:]

def iterate_n_dir(folder: Path, group_exp: bool=False, n_max=1000) -> Path:
	if not group_exp and Path(folder).is_dir():
		if not re.match(folder.name, '-[0-9]*'):
			folder = add_to_Path(folder, '-0')
		for i in range(n_max+1):
			folder = folder.parent / folder.name.split('-')[0]
			folder = add_to_Path(folder, f'-{i}')
			if not folder.exists():
				break   
	return folder

### do things

def mkdir(path: Path) -> Path:
	path = Path(path)
	if path.suffix != '':
		path = path.parent
	try:
		if not path.exists() or not path.is_dir():
			path.mkdir(parents=True)
	except Exception as e:
		print(e)
	return path

def add_to_Path(path: Path, string: Union[str, Path]):
	suffix = path.suffix
	path = path.with_suffix('')
	return Path(str(path) + str(string)).with_suffix(suffix)

### convert things

def dict_to_cmd(d: dict):
	items = d.items()
	items = ((k, (v.tolist() if isinstance(v, np.ndarray) else v)) for (k,v) in items)
	items = ((str(k).replace(" ", ""), str(v).replace(" ", "")) for (k,v) in items)
	return ' '.join([f'--{k} {v}' for k,v in items if v])

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
	
def cmd_to_dict(cmd:Union[str, list], ref:dict, delim:str=' --', d=None):
	"""
	fmt: [--flag, arg, --true_flag, --flag, arg1]
	# all flags double dash because of negative numbers duh """
	
	cmd = ' ' + (' '.join(cmd) if isinstance(cmd, list) else cmd)  # add initial space in case single flag
	cmd = [x.strip() for x in cmd.split(delim)][1:]
	cmd = [[sub.strip() for sub in x.split('=', maxsplit=1)] 
		   if '=' in x else 
		   [sub.strip() for sub in x.split(' ', maxsplit=1)] 
		   for x in cmd]
	[x.append('True') for x in cmd if len(x)==1]
	
	d = dict()
	for k,v in cmd:
		v = format_cmd_item(v)
		k = k.replace(' ', '')
		v_ref = ref.get(k, None)
		if v_ref is None:
			print(f'{k} not in ref')
		d[k] = type_me(v, v_ref, is_cmd_item=True)
	return d

def npify(v):
	return torch.tensor(v.numpy())

def format_cmd_item(v):
	v = v.replace('(', '[').replace(')', ']')
	return v.replace(' ', '')
	
def type_me(v, v_ref=None, is_cmd_item=False):
	def count_leading_char(s, char): 
		# space=r"^\s*" bracket=r'^[*'
		match = re.search(rf'^{char}*', s)
		return 0 if not match else match.end()
	
	if is_cmd_item:
		""" Accepted: 
		bool, list of list (str, float, int), dictionary, str, explicit str (' "this" '), """
		v = format_cmd_item(v)
		
		if v.startswith('[['):
			v = v.strip('[]')
			nest_lst = v.split('],[')
			return [type_me('['+lst+']', v_ref[0], is_cmd_item=True) for lst in nest_lst]
		
		if v.startswith('['):
			v = v.strip('[]')
			v = v.split(',')
			return [type_me(x, v_ref[0]) for x in v]
		
		booleans = ['True', 'true', 't', 'False', 'false', 'f']
		if v in booleans: 
			return booleans.index(v) < 3  # 0-2 True 3-5 False
	
	if v_ref is not None:
		type_ref = type(v_ref)
		if isinstance(v, str):
			v = v.strip('\'\"')
		
		if isinstance(v, (np.ndarray, np.generic)):
			if isinstance(v.flatten()[0], str):
				return v.tolist()
			return v
		
		if isinstance(v, list):
			if isinstance(flat_any(v)[0], str):
				return v
			return np.asarray(v)
		
		if isinstance(v, torch.Tensor):
			return v

		return type_ref(v)
		
	try:
		return literal_eval(v)
	except:
		return str(v).strip('\'\"')
	
### run things

def run_cmds(cmd:Union[str, list], cwd:Union[str, Path]='.', silent=False, _res=[]):
	for cmd_1 in (cmd if isinstance(cmd, list) else [cmd]): 
		try:
			cmd_1 = [c.strip() for c in cmd_1.split(' ')]
			_res = subprocess.run(cmd_1, cwd=str(cwd), capture_output=True, text=True)
			if not silent:
				print(f'Run: {cmd_1} at {cwd}')
				print('stdout:', _res.stdout.replace("\n", " "), 'stderr:', _res.stderr.replace("\n", ";"))
		except Exception as e:
			if not silent:
				print(cmd_1, e)
			return ('Fail', '')
	return _res.stdout.rstrip('\n')


def run_cmds_server(server:str, user:str, cmd:Union[str, list], cwd=Union[str, Path], _res=[]):
	client = paramiko.SSHClient()    
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # if not known host
	client.connect(server, username=user)
	for cmd_1 in (cmd if isinstance(cmd, list) else [cmd]):
		print(f'Remote run: {cmd_1} at {user}@{server}:{cwd}')
		stdin, stdout, stderr = client.exec_command(f'cd {str(cwd)}; {cmd_1}')
		stderr = '\n'.join(stderr.readlines())
		stdout = '\n'.join(stdout.readlines())
		print('stdout:', stdout, 'stderr:', stderr)
	client.close()
	return stdout.replace("\n", " ")

### flatten things

def flat_arr(v):
	return v.reshape(v.shape[0], -1)

def flat_list(lst):
	items = []
	for v in lst:
		if isinstance(v, list):
			items.extend(flat_list(v))
		else:
			items += [v]
	return items

def flat_dict(d:dict):
	items = []
	for k,v in d.items():
		if isinstance(v, dict):
			items.extend(flat_dict(v).items())
		else:
			items.append((k, v))
	return dict(items)

def flat_any(v: Union[list, dict]):
	if isinstance(v, list):
		return flat_list(v)
	if isinstance(v, dict):
		return flat_dict(v)


### wandb ###

def dict_to_wandb(d:dict, parent='', sep='.', items:list=None)->dict:
	items = items or []
	for k, v in d.items():
		if callable(v):
			continue
		if isinstance(v, Path):
			parent = 'path'
		name = parent + sep + k if parent else k
		if isinstance(v, dict):
			items.extend(dict_to_wandb(v, name, items=items).items())
		items.append((name, v))
	return dict(items)


try:

	def compute_metrix(d:dict, mode='tr', fancy=None, ignore = [], _d = {}):
		
		for k,v in d.copy().items():
			if any([ig in k for ig in ignore+['step',]]):
				continue 
			
			if not fancy is None:
				k = fancy.get(k, k)

			v_mean = optree.tree_map(lambda x: x.mean() if not np.isscalar(x) else x, v)  if not np.isscalar(v) else v
			v_std = optree.tree_map(lambda x: x.std() if not np.isscalar(x) else x, v)  if not np.isscalar(v) else v
			
			group = mode
			if 'grad' in k:
				group = mode + '/grad'
			elif 'param' in k:
				group += '/param'
				
			_d = collect_stats(k, v_mean, _d, p=group, suf=r'_\mu$')
			_d = collect_stats(k, v_std, _d, p=group+'/std', suf=r'_\sigma$')

		return _d

except: 
	print('Metrix: No Torch')

def torchify_tree(v: np.ndarray, v_ref: torch.Tensor):
	leaves, tree_spec = optree.tree_flatten(v)
	leaves_ref, _ = optree.tree_flatten(v_ref)
	leaves = [torch.tensor(data=v, device=ref.device, dtype=ref.dtype) 
		   	  if isinstance(ref, torch.Tensor) else v 
			  for v, ref in zip(leaves, leaves_ref)]
	return optree.tree_unflatten(treespec=tree_spec, leaves=leaves)

def numpify_tree(v: torch.Tensor):
	leaves, treespec = optree.tree_flatten(v)
	leaves = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in leaves]
	return optree.tree_unflatten(treespec=treespec, leaves=leaves)

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
		job_type:		str		= 'debug'		
		wb_mode: 		str		= 'disabled'
		wb_sweep: 		bool	= False
  
		entity:			str		= property(lambda _: _._p.project)
		program: 		Path	= property(lambda _: Path( _._p.project_dir, _._p.run_name))
		sweep_path_id:  str     = property(lambda _: f'{_.entity}/{_._p.project}/{_._p.exp_name}')
		wb_type: 		str		= property(lambda _: _.wb_sweep*'sweeps' or _._p.sweep.run_sweep*f'groups' or 'runs')
		run_url: 		str		= property(lambda _: f'www.wandb.ai/{_.entity}/{_._p.project}/{_.wb_type}/{_._p.exp_name}')
  
	class distribute(Sub):
		head:			bool	= True 
		dist_mode: 		str		= 'pyfig'  # accelerate
		dist_id:		str		= ''
		sync_step:		int		= 5

		gpu_id_cmd:		str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
		gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_.gpu_id_cmd, silent=True)).split('.')[0])
		dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])

	class resource(Sub):
		submit: 		bool	= False
		cluster_submit: Callable= None
		script:			Callable= None

	class git(Sub):
		branch:     str     = 'main'
		remote:     str     = 'origin' 

		commit_id_cmd:	str 	= 'git log --pretty=format:%h -n 1'
		commit_id:   	list	= property(lambda _: run_cmds(_.commit_id_cmd, cwd=_._p.project_dir, silent=True))
		# commit_cmd:	str     = 'git commit -a -m "run"' # !NB no spaces in msg 
		# commit: 		list	= property(lambda _: run_cmds(_.commit_cmd, cwd=_.project_dir)) 
		# pull_cmd:		str 	= ['git fetch --all', 'git reset --hard origin/main']
		# pull:   		list	= property(lambda _: run_cmds(_.pull_cmd, cwd=_.project_dir))
  
	home:				Path	= Path().home()
	dump:               Path    = property(lambda _: Path('dump'))
	dump_exp_dir: 		Path 	= property(lambda _: _.dump/'exp')
	tmp_dir:            Path	= property(lambda _: _.dump/'tmp')
	project_dir:        Path    = property(lambda _: _.home / 'projects' / _.project)
	cluster_dir: 	Path    = property(lambda _: Path(_.exp_dir, 'cluster'))
	exchange_dir: 	Path    = property(lambda _: Path(_.exp_dir, 'exchange'))

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
		""" Steps
		0- subclasses inherit from personal bases
		1- initialised subclasses, because they don't have properties otherwise
		"""

		for sub_name, sub in ii.sub_cls.items():
			setattr(ii, sub_name, sub(parent=ii))

		c_init = flat_any(ii.d)
		sys_arg = sys.argv[1:]
		sys_arg = cmd_to_dict(sys_arg, c_init) if not notebook else {}  
		init_arg = flat_any(init_arg) | (sweep or {})

		ii.update_configuration(init_arg | sys_arg)

		ii.setup_exp_dir(group_exp=ii.group_exp, force_new_id=False)

		ii.debug_log([dict(os.environ.items()), ii.d,], [ii.env_log_path, ii.d_log_path])
  
		if not ii.resource.submit and ii.distribute.head:
			run = wandb.init(
				project     = ii.project, 
				group		= ii.exp_name,
				id          = ii.exp_id,
				dir         = ii.exp_dir,
				entity      = ii.wb.entity,  	
				mode        = ii.wb.wb_mode,
				config      = dict_to_wandb(ii.d),
			)
		
		if ii.resource.submit:

			run_or_sweep_d = ii.get_run_or_sweep_d()

			for i, run_d in enumerate(run_or_sweep_d):
				is_first_run = i == 0

				group_exp = ii.group_exp or (is_first_run and ii.sweep.run_sweep)
				ii.setup_exp_dir(group_exp= group_exp, force_new_id= True)
				
				base_d = inst_to_dict(ii, attr=True, sub_cls=True, flat=True, ignore=ii.ignore, debug=ii.debug)
				run_d = base_d | run_d

				if is_first_run:
					ii.debug_log([dict(os.environ.items()), run_d], [ii.env_log_path, ii.d_log_path])
				
				ii.resource.cluster_submit(run_d)

			sys.exit(ii.wb.run_url)

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
		if ii.exp_dir and not force_new_id:
			return None
		exp_name = ii.exp_name or 'junk'
		exp_group_dir = Path(ii.dump_exp_dir, 'sweep'*ii.sweep.run_sweep, exp_name)
		exp_group_dir = iterate_n_dir(exp_group_dir, group_exp=group_exp)
		ii.exp_name = exp_group_dir.name
		ii.exp_id = (not force_new_id)*ii.exp_id or gen_time_id(7)
		ii.exp_dir = exp_group_dir/ii.exp_id
		[mkdir(ii.exp_dir/_dir) for _dir in ['cluster', 'exchange', 'wandb']]
  
	def get_run_or_sweep_d(ii,):
		ii.resource.submit = False
		
		if not ii.sweep.run_sweep:
			""" single run
			takes configuration from base in submit loop
			"""
			return [dict(),] 

		d = deepcopy(ii.sweep.d)
		sweep_keys = list(d['parameters'].keys())
		sweep_vals = [v['values'] for v in d['parameters'].values()]
		sweep_vals = get_cartesian_product(*sweep_vals)
		print(f'### sweep over {sweep_keys} ({len(sweep_vals)} total) ###')
		return [{k:v for k,v in zip(sweep_keys, v_set)} for v_set in sweep_vals]

	def debug_log(ii, d_all:list, p_all: list):
		for d, p in zip(d_all, p_all):
			ii.log(d, path=p)
			
	def partial(ii, f:Callable, args=None, **kw):
		d = flat_any(args if args else ii.d)
		d_k = inspect.signature(f.__init__).parameters.keys()
		d = {k:copy(v) for k,v in d.items() if k in d_k} | kw
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
	
	def sync(ii, step: int, v_tr: dict):
		v_sync = numpify_tree(v_tr)
		try:
			gc.disable()
			v_path = (ii.exchange_dir / f'{step}_{ii.distribute.dist_id}').with_suffix('.pk')
			v, treespec = optree.tree_flatten(deepcopy(v_tr))
			dump(v_path, v)
		except Exception as e:
			print(e)
		finally:
			gc.enable()
		
		if ii.distribute.head:
			### 1 wait for workers to dump ###
			n_ready = 0
			while n_ready < ii.resource.n_gpu:
				k_path_all = list(ii.exchange_dir.glob(f'{step}_*')) 
				n_ready = len(k_path_all)
				sleep(0.02)

			### 2 collect arrays ###
			try:
				gc.disable()
				leaves = []
				for p in k_path_all:
					v_dist_i = load(p)
					leaves += [v_dist_i]
			except Exception as e:
				print(e)
			finally:
				gc.enable()

			### 3 mean arrays ###
			v_mean = [np.stack(leaves).mean(axis=0) for leaves in zip(*leaves)]

			try:
				gc.disable()
				for p in k_path_all:
					dump(add_to_Path(p, '-mean'), v_mean)
			except Exception as e:
				print(e)
			finally:
				for p in k_path_all:
					p.unlink()
				gc.enable()

		v_mean_path = add_to_Path(v_path, '-mean')
		while v_path.exists():
			while not v_mean_path.exists():
				sleep(0.02)
			sleep(0.02)
		try:
			gc.disable()
			v_sync = load(v_mean_path)  # Speed: Only load sync vars
			v_sync = optree.tree_unflatten(treespec=treespec, leaves=v_sync)
		except Exception as e:
			print(e)
			gc.enable()
			v_sync = v_tr
		finally: # ALWAYS EXECUTED
			v_mean_path.unlink()
			gc.enable()
			v_sync = torchify_tree(v_sync, v_tr)
			return v_sync

	@staticmethod
	def log(info: Union[dict,str], path='dump/tmp/log.tmp'):
		mkdir(path)
		info = pprint.pformat(info)
		with open(path, 'w') as f:
			f.writelines(info)

### slurm things

from simple_slurm import Slurm

class niflheim_resource(Sub):
	env: str     	= ''
	n_gpu: int 		= 1
	
	architecture:   str 	= 'cuda'
	nifl_gpu_per_node: int  = property(lambda _: 10)

	job_id: 		str  	= property(lambda _: os.environ.get('SLURM_JOBID', 'No SLURM_JOBID available.'))  # slurm only
	hostname: 		str 	= os.environ['HOSTNAME'] # shell restriction unknown

	pci_id_cmd:		str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
	pci_id:			str		= property(lambda _: ''.join(run_cmds(_.pci_id_cmd, silent=True)))

	n_device_env:	str		= 'CUDA_VISIBLE_DEVICES'
	n_device:       int     = property(lambda _: sum(c.isdigit() for c in os.environ.get(_.n_device_env, 'ZERO')))

	### Slurm Configuration ###
	export			= 'ALL'
	nodes           = '1' 			# (MIN-MAX) 
	mem_per_cpu     = 1024
	cpus_per_task   = 4
	partition       = 'sm3090'
	time            = '0-01:00:00'  # D-HH:MM:SS

	gres            = property(lambda _: 'gpu:RTX3090:' + str(_.n_gpu))
	ntasks          = property(lambda _: _.n_gpu)
	job_name        = property(lambda _: _._p.exp_name)
	output          = property(lambda _: _._p.cluster_dir/'o-%j.out')
	error           = property(lambda _: _._p.cluster_dir/'e-%j.err')
 
	# n_running_cmd:	str		= 'squeue -u amawi -t pending,running -h -r'
	# n_running:		int		= property(lambda _: len(run_cmds(_.n_running_cmd, silent=True).split('\n')))	
	# running_max: 	int     = 20
 
	def script(ii, job: dict):

		mod = ['module purge', 'module load foss', 'module load CUDA/11.7.0']
		env = ['source ~/.bashrc', f'conda activate {ii.env}',]
		export = ['export $SLURM_JOB_ID',]
		debug = ['echo $SLURM_JOB_GPUS', 'echo $cluster_JOB_NODELIST', 'nvidia-smi']
		srun_cmd = 'srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=1024 --ntasks=1 --exclusive --label '
		body = mod + env + debug
 
		for i in range(ii.n_gpu):
			
			device_log_path = ii._p.cluster_dir/(str(i)+"_device.log") # + ii._p.hostname.split('.')[0])
		
			cmd = dict_to_cmd(job)
   
			cmd = f'python -u {job["run_name"]} {cmd}'
				
			body += [f'{srun_cmd} {cmd} 1> {device_log_path} 2>&1 & ']
   
		body += ['wait',]
		return '\n'.join(body)

	def cluster_submit(ii, job):
		sbatch = ii.script(job)
		slurm = Slurm(
			export			= ii.export,
			nodes           = ii.nodes        ,
			mem_per_cpu     = ii.mem_per_cpu  ,
			cpus_per_task   = ii.cpus_per_task,
			partition       = ii.partition    ,
			time            = ii.time         ,
			gres            = ii.gres         ,
			ntasks          = ii.ntasks       ,
			job_name        = ii.job_name     ,
			output          = ii.output       ,
			error           = ii.error        ,
		)
		slurm.sbatch(sbatch)


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
 
# try:
#     from flax.core.frozen_dict import FrozenDict
#     from jax import random as rnd
	
#     def gen_rng(rng, n_device):
#         """ rng generator for multi-gpu experiments """
#         rng, rng_p = torch.split(rnd.split(rng, n_device+1), [1,])
#         return rng.squeeze(), rng_p	
		

#     def compute_metrix(d:dict, mode='tr', fancy=None, ignore = [], _d = {}):
		
#         for k,v in d.items():
#             if any([ig in k for ig in ignore+['step']]):
#                 continue 
			
#             if not fancy is None:
#                 k = fancy.get(k, k)

#             v = jax.device_get(v)
			
#             if isinstance(v, FrozenDict):
#                 v = v.unfreeze()
			
#             v_mean = jax.tree_map(lambda x: x.mean(), v) if not np.isscalar(v) else v
#             v_std = jax.tree_map(lambda x: x.std(), v) if not np.isscalar(v) else 0.

#             group = mode
#             if 'grad' in k:
#                 group = mode + '/grad'
#             elif 'param' in k:
#                 group += '/param'
				
#             _d = collect_stats(k, v_mean, _d, p=group, suf=r'_\mu$')
#             _d = collect_stats(k, v_std, _d, p=group+'/std', suf=r'_\sigma$')

#         return _d

#     ### type testing ### 

#     def test_print_fp16_no_cast():
#         x = torch.ones([1], dtype='float16')
#         print(x)  # FAILS

#     def test_print_fp16():
#         x = torch.ones([1], dtype='float16')
#         x = x.astype('float16')
#         print(x)  # OK

#     def test_print_fp32():
#         x = torch.ones([1], dtype='float16')
#         x = x.astype('float16')
#         x = x.astype('float32')
#         print(x)  # OK

#     def test_print_fp32_to_fp16_cast():
#         x = torch.ones([1], dtype='float32')
#         x = x.astype('float16')
#         print(x)  # FAILS

# except:
#     print('no flax or jax installed')

""" BONE ZONE
# def to_cmd(d:dict):
#     ' Accepted: int, float, str, list, dict, np.ndarray'
	
#     def prep_cmd_item(k:str, v:Any):
#         if isinstance(v, np.ndarray):
#             v = v.tolist()
#         return str(k).replace(" ", ""), str(v).replace(" ", "")
	
#     return dict(prep_cmd_item(k,v) for k,v in d.items())
# 
# 
"""