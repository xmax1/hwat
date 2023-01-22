import sys
from pyfig_utils import PyfigBase, Param, lo_ve
import optuna
from typing import Callable
from time import sleep
import pprint
import os
from utils import debug_dict

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
		return lambda : trial.suggest_float(name, *v.domain, log=v.log)
	
	raise Exception(f'{v} not supported in hypam opt')
 
def get_hypam_from_study(trial: optuna.Trial, sweep: dict) -> dict:
	print('trialing hypam:sweep')
	debug_dict(d=sweep, msg='get_hypam_from_study')
	for i, (name, v) in enumerate(sweep.items()):
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

	debug_dict(d=study, msg='study')
	debug_dict(d=study.trials, msg='trials')

	return dict(c_update=study.best_params)