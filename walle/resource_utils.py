


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
import torch 
from functools import partial 
import pickle as pk
import yaml
import json
from functools import partial
import numpy as np
from .utils import get_max_n_from_filename

from .utils import dict_to_cmd, cmd_to_dict, dict_to_wandb, debug_dict
from .utils import mkdir, iterate_n_dir, gen_time_id, add_to_Path, dump, load
from .utils import get_cartesian_product, type_me, run_cmds, flat_any 

from torch import nn

this_dir = Path(__file__).parent
hostname = os.environ['HOSTNAME']

from .pyfig_utils import PlugIn, PyfigBase


class niflheim(PyfigBase.resource):
	_p: PyfigBase   = None

	env: str     	= ''
	n_gpu: int 		= 1
	n_node: int		= 1

	architecture:   str 	= 'cuda'
	nifl_gpu_per_node: int  = property(lambda _: 10)
	device_log_path: str 	= '' 

	job_id: 		str  	= property(lambda _: os.environ.get('SLURM_JOBID', 'No SLURM_JOBID available.'))  # slurm only

	_pci_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
	pci_id:			str		= property(lambda _: ''.join(run_cmds(_._pci_id_cmd, silent=True)))
	gpu_i: 			int		= 0

	n_device_env:	str		= 'CUDA_VISIBLE_DEVICES'
	# n_device:       int     = property(lambda _: sum(c.isdigit() for c in os.environ.get(_.n_device_env, '')))
	n_device:       int     = property(lambda _: len(os.environ.get(_.n_device_env, '').replace(',', '')))

	class slurm_c(PlugIn):
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

		# module load foss
		body = []
		body = f"""
		module purge
		source ~/.bashrc
  		conda activate {ii.env}
		export exp_id="{job["exp_id"]}"
		echo exp_id-${{exp_id}}
		MKL_THREADING_LAYER=GNU   
		export ${{MKL_THREADING_LAYER}}
		"""
		# important, here, hf_accelerate and numpy issues https://github.com/pytorch/pytorch/issues/37377

		extra = """
		module load CUDA/11.7.0
		module load OpenMPI
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

		body += extra
		# body += debug_body


		if ii._p.wb.wb_sweep:
			body += [f'wandb controller {ii._p.wb.sweep_id}'] # {ii._p.wb.sweep_path_id}
			body += [f'wandb agent {ii._p.wb.sweep_id} 1> {ii.device_log_path(rank=0)} 2>&1 ']
			body += ['wait',]

		elif ii._p.dist.dist_method == 'hf_accelerate':
			print('\n accelerate distribution')
			# ! backslash must come between run.py and cmd
			cmd = dict_to_cmd(job, exclude_none=True)
			body += f'{ii._p.dist._launch_cmd} {job["run_name"]} \ {cmd} 1> {ii.device_log_path(rank=0)} 2>&1 \n'  
   
		elif ii._p.dist.dist_method == 'naive':
			print('\n pyfig distribution')
			for i in range(ii.n_gpu):
				job.update(dict(head= i==0, gpu_i=i))
				cmd = dict_to_cmd(job)
				cmd = f'python -u {job["run_name"]} {cmd}'
				body += f'\n{ii._p.dist._launch_cmd} {cmd} 1> {ii.device_log_path(rank=i)} 2>&1 & \n'
			body += '\nwait \n'
		
		body += '\necho End \n'

		if ii._p.debug:
			print(body)
		
		body = body.split('\n')
		body = [b.strip() for b in body]
		body = '\n'.join(body)
		
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

