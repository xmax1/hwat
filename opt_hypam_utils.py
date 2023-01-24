
from pathlib import Path
import sys
from pyfig_utils import PyfigBase, Param, lo_ve
import optuna
from typing import Callable
from time import sleep
import pprint
import os
from utils import debug_dict
from pyfig import Pyfig
import numpy as np
from optuna import Trial

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
	max_lr		=	Param(values=[0.1, 0.01, 0.001], dtype=str, conditional=dict(opt_name='AdaHessian')),
	ke_method	=	Param(values=['ke_grad_grad_method', 'ke_vjp_method',  'ke_jvp_method'], dtype=str),
	n_step		=	Param(values=[1000, 2000, 4000], dtype=int),

)


"""

def str_lower_eq(a: str, b:str):
	return a.lower()==b.lower()

def objective(trial: Trial, c: Pyfig, run: Callable):
	c_update = get_hypam_from_study(trial, c.sweep.parameters)
	c.mode = 'train'
	v_tr = run(c=c, init_d=c_update)
	c.mode = 'eval'
	v_eval = run(c=c, **v_tr)
	return np.stack(v_eval['opt_obj_all']).mean()


def suggest_hypam(trial: optuna.Trial, name: str, v: Param):
	if isinstance(v, dict):
		v = Param(**v)
	debug_dict(d=v.d, msg='suggest_hypam:Param')

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

def order_conditionals(sweep_p: dict, order: list=None):
	from collections import OrderedDict
	wait = []
	order = []
	for name, v in sweep_p.items():
		if v.conditional:
			if any([k_cond in order for k_cond in v.conditional]):
				order += [name,]
			else:
				wait += [(name, v),]
		else:
			order += [name,]
	if wait:
		wait = OrderedDict(reversed(wait))
		order += order_conditionals(wait, order)
	return order

def get_hypam_from_study(trial: optuna.Trial, sweep_p: dict) -> dict:
	print('trialing hypam:sweep')
	debug_dict(d=sweep_p, msg='get_hypam_from_study')
	sweep_p_order = order_conditionals(sweep_p)
	for i, name in enumerate(sweep_p_order):
		v = sweep_p[name]
		v = suggest_hypam(trial, name, v)
		c_update = {name:v} if i==0 else {**c_update, name:v}
	debug_dict(d=c_update, msg='get_hypam_from_study:c_update')
	return c_update


def opt_hypam(objective: Callable, c: PyfigBase):
	print('hypam opt create/get study')
 
	if c.distribute.head:
		study = optuna.create_study(
			study_name		= c.sweep.sweep_name,
			load_if_exists 	= True, 
			direction 		= "minimize",
			storage			= c.sweep.storage,
			sampler 		= lo_ve(c.exp_dir/'sampler.pk') or optuna.samplers.TPESampler(seed=c.seed),
			pruner			= optuna.pruners.MedianPruner(n_warmup_steps=10),
		)
	else:
		while not c.sweep.storage.exists():
			print('waiting for opt storage...')
			sleep(3)

	study.optimize(
		objective, 
		n_trials=c.sweep.n_trials, 
		timeout=None, 
		callbacks=None, 
		show_progress_bar=True, 
		gc_after_trial=True
	)

	best_param = dict(c_update=study.best_params)
	print(Path('.'), '\nstudy:best_param = \n', best_param)
	debug_dict(d=study, msg='study')
	debug_dict(d=study.trials, msg='trials')

	return 