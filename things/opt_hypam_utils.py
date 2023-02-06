
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

import torch 

def objective(trial: Trial, run_trial: Callable, c: PyfigBase):
	
	print('trial: ', trial.number)
	c_update = get_hypam_from_study(trial, c.sweep.parameters)
	pprint.pprint(c_update)

	try:
		v_run: dict = run_trial(c= c, c_update_trial= c_update)
	except torch.cuda.OutOfMemoryError as e:
		print('trial out of memory')
		return float("inf")

	dummy = [np.array([0.0]), np.array([0.0])]
	opt_obj_all = v_run.get(c.tag.v_cpu_d, {}).get(c.tag.opt_obj_all, dummy)
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

from copy import deepcopy
from .pyfig_utils import Param

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

import time


def opt_hypam(c: PyfigBase, run_trial: Callable):
	print('opt_hypam:create_study rank,head,is_logging_process', c.dist.rank, c.dist.head, c.is_logging_process)

	if c.dist.rank:
		print('opt_hypam:waiting_for_storage rank,head,is_logging_process', c.dist.rank, c.dist.head, c.is_logging_process)
		while not len(list(c.exp_dir.glob('*.db'))):
			sleep(5.)
		sleep(5.)
		study = optuna.load_study(study_name= c.sweep.sweep_name, storage=c.sweep.storage)

	else:
		print('opt_hypam:creating_study rank,head', c.dist.rank, c.dist.head)
		study = optuna.create_study(
			direction 		= "minimize",
			study_name		= c.sweep.sweep_name,
			storage			= c.sweep.storage,
			sampler 		= lo_ve(path=c.exp_dir/'sampler.pk') or optuna.samplers.TPESampler(seed=c.seed),
			pruner			= optuna.pruners.MedianPruner(n_warmup_steps=10),
			load_if_exists 	= True, 
		)

	from optuna.study import MaxTrialsCallback
	from optuna.trial import TrialState
	from functools import partial
	import json

	_objective = partial(objective, c=c, run_trial=run_trial)
	study.optimize(
		_objective, 
		n_trials=c.sweep.n_trials, 
		timeout=None, 
		callbacks=[MaxTrialsCallback(c.sweep.n_trials, states=(TrialState.COMPLETE,))],
		gc_after_trial=True
	)

	v_run = dict(c_update= study.best_params)
	path = c.exp_dir/'best_params.json'
	path.write_text(json.dumps(study.best_params, indent=4))
	print('\nstudy:best_params')
	pprint.pprint(v_run)
	return v_run