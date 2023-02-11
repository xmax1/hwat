
from copy import deepcopy
from pathlib import Path
from typing import Callable
from time import sleep
import pprint
import numpy as np

from .utils import PlugIn
from .core import lo_ve, get_cartesian_product

from functools import partial
import json



"""
- print cmd to join run

python run.py --submit --multimode opt_hypam
--n_b ?? 
--record 

--exp_id
--exp_name

parameters: 	dict 	= dict(

	lr			=	Param(domain=(0.0001, 1.), log=True),
	opt_name	=	Param(values=['AdaHessian',  'RAdam'], dtype=str),
	max_lr		=	Param(values=[0.1, 0.01, 0.001], dtype=str, condition=dict(opt_name='AdaHessian')),
	ke_method	=	Param(values=['ke_grad_grad_method', 'ke_vjp_method',  'ke_jvp_method'], dtype=str),
	n_step		=	Param(values=[1000, 2000, 4000], dtype=int),

)


"""

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
	
	def get(ii, key, dummy=None):
		return getattr(ii, key, dummy)

def str_lower_eq(a: str, b:str):
	return a.lower()==b.lower()

class SweepBase(PlugIn):
	sweep_name: 	str		= 'sweep'	
	parameters: 	dict 	= dict(
		# dtype			=	Param(values=[torch.float32, torch.float64], dtype=str), # !! will not work
		example			=	Param(values=['2',  '1'], dtype=str),
		another_one		= 	Param(domain=[0.01, 1.], dtype=float, condition=['example',]),
	)

	def get_sweep(ii,):
		params: dict[str,Param] = ii.parameters
		sweep_keys = list(params.keys())
		sweep_vals = [v.get('values', []) for v in params.values()]
		if any([v.domain is not None for v in params.values()]):
			raise Exception('domain no esta por sweep, use opt_hypam')
		sweep_vals = get_cartesian_product(*sweep_vals)
		print(f'### sweep over {sweep_keys} ({len(sweep_vals)} total) ###')
		return [{k:v for k,v in zip(sweep_keys, v_set)} for v_set in sweep_vals]


from .core import TryImportThis

with TryImportThis('optuna') as _optuna:
	
	from optuna import Trial
	from optuna.study import MaxTrialsCallback
	from optuna.trial import TrialState
	import optuna
	from functools import partial
	import json


	def objective(trial: Trial, run_trial: Callable, c):
		
		print('trial: ', trial.number)
		c_update = get_hypam_from_study(trial, c.sweep.parameters)
		pprint.pprint(c_update)

		try:
			v_run: dict = run_trial(c= c, c_update_trial= c_update)
		except RuntimeError as e:
			print('trial out of memory', e)
			return float("inf")

		dummy = [np.array([0.0]), np.array([0.0])]
		opt_obj_all = v_run.get(c.v_cpu_d_tag, {}).get(c.opt_obj_all_tag, dummy)
		return np.asarray(opt_obj_all).mean()


	def suggest_hypam(trial: optuna.Trial, name: str, v: Param):

		if isinstance(v, dict):
			v = Param(**v)

		if not v.domain:
			return trial.suggest_categorical(name, v.values)

		if v.sample:
			if v.step_size:
				return trial.suggest_discrete_uniform(name, *v.domain, q=v.step_size)
			elif v.log:
				return trial.suggest_loguniform(name, *v.domain)
			else:
				return trial.suggest_uniform(name, *v.domain)

		variables = v.values or v.domain
		dtype = v.dtype or type(variables[0])
		if dtype is int:
			return trial.suggest_int(name, *v.domain, log=v.log)
		
		if dtype is float:
			return trial.suggest_float(name, *v.domain, log=v.log)
		
		raise Exception(f'{v} not supported in hypam opt')


	def get_hypam_from_study(trial: optuna.Trial, sweep_p: dict[str, Param]) -> dict:

		c_update = {}
		for name, param in sweep_p.items():
			v = suggest_hypam(trial, name, param)
			c_update[name] = v
		
		for k, v in deepcopy(c_update).items():
			condition = sweep_p[k].condition
			if condition:
				if not any([cond in c_update.values() for cond in condition]):
					c_update.pop(k)

		print('optuna:get_hypam_from_study \n')
		pprint.pprint(c_update)
		return c_update

	class Optuna(SweepBase):
		# globals

		# plugin_vars 
		n_trials: 		int		= 20
		storage: 		Path = property(lambda _: 'sqlite:///'+str(_.p.paths.exp_dir / 'hypam_opt.db'))

		def opt_hypam(ii, run_trial: Callable):
			print('opt_hypam:create_study rank,head,is_logging_process', ii.p.dist.rank, ii.p.dist.head, ii.p.is_logging_process)

			if ii.p.dist.rank:
				print('opt_hypam:waiting_for_storage rank,head,is_logging_process', ii.p.dist.rank, ii.p.dist.head, ii.p.is_logging_process)
				while not len(list(ii.p.paths.exp_dir.glob('*.db'))):
					sleep(5.)
				sleep(5.)
				study = optuna.load_study(study_name= ii.sweep_name, storage= ii.storage)

			else:
				print('opt_hypam:creating_study rank,head', ii.p.dist.rank, ii.p.dist.head)
				study = optuna.create_study(
					direction 		= "minimize",
					study_name		= ii.sweep_name,
					storage			= ii.storage,
					sampler 		= lo_ve(path=ii.p.paths.exp_dir/'sampler.pk') or optuna.samplers.TPESampler(seed= ii.p.seed + ii.p.dist.rank),
					pruner			= optuna.pruners.MedianPruner(n_warmup_steps=10),
					load_if_exists 	= True, 
				)

			_objective = partial(objective, run_trial= run_trial, c= ii.p)
			study.optimize(
				_objective, 
				n_trials= ii.n_trials, 
				timeout=None, 
				callbacks=[MaxTrialsCallback(ii.n_trials, states=(TrialState.COMPLETE,))],
				gc_after_trial=True
			)

  
			v_run = dict(c_update= study.best_params)
			path = ii.p.paths.exp_dir/'best_params.json'
			path.write_text(json.dumps(study.best_params, indent=4))
			print('\nstudy:best_params')
			_ = ii.p.dist.sync(dict(test= study.best_params), sweep_method= ii.p.gather_tag) # ugly. wait for processes. 
			pprint.pprint(v_run)
			return v_run




"""

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
 
 
 """