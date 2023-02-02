
from pathlib import Path
import sys
import optuna
from typing import Callable
from time import sleep
import pprint
import os
import numpy as np
from optuna import Trial

from .utils import debug_dict
from .pyfig_utils import PyfigBase, Param, lo_ve
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

def str_lower_eq(a: str, b:str):
	return a.lower()==b.lower()

def objective(trial: Trial, c: PyfigBase, run: Callable):
	c_update_next = get_hypam_from_study(trial, c.sweep.parameters)
	print('trial')
	pprint.pprint(c_update_next)
	c.mode = 'train'
	v_run = run(c=c, c_update=c_update_next)
	c.mode = 'eval-dark'
	v_run = run(c=c, v_init=v_run.get('v_init_next', {}))
	return np.stack(v_run['opt_obj_all']).mean()


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

from copy import deepcopy

def get_hypam_from_study(trial: optuna.Trial, sweep_p: dict) -> dict:

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

import time

def opt_hypam(objective: Callable, c: PyfigBase):
	print('hypam opt create/get study')
 
	# how to id the different gpus give rank is 0 for 
	
	if not c.dist.head:
		while not len(list(c.exp_dir.glob('*.db'))):
			print('waiting for opt storage...')
			sleep(5.)
		sleep(c.dist.rank)
	
		study = optuna.load_study(study_name=c.sweep.sweep_name, storage=c.sweep.storage)
	else:
		study = optuna.create_study(
			direction 		= "minimize",
			study_name		= c.sweep.sweep_name,
			load_if_exists 	= True, 
			storage			= c.sweep.storage,
			sampler 		= lo_ve(path=c.exp_dir/'sampler.pk') or optuna.samplers.TPESampler(seed=c.dist.rank),
			pruner			= optuna.pruners.MedianPruner(n_warmup_steps=10),
		)

	study.optimize(
		objective, 
		n_trials=c.sweep.n_trials, 
		timeout=None, 
		callbacks=None,
		gc_after_trial=True
	)

	best_param = dict(c_update=study.best_params)
	print(Path('.'), '\nstudy:best_param = \n', best_param)
	debug_dict(d=study, msg='study')
	debug_dict(d=study.trials, msg='trials')

	return 