import sys
from typing import Callable
from pathlib import Path
import numpy as np
import os
from pathlib import Path

import wandb

from utils import cmd_to_dict, dict_to_wandb, flat_any, dict_to_cmd, run_cmds, cls_to_dict
from utils import PyfigBase, Sub, niflheim_resource, hostname

from user_secret import user


class Pyfig(PyfigBase):

	user: 				str 	= user
 
	project:            str     = 'hwat'
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
  
		entity:			str		= property(lambda _: _.p.project)
		program: 		Path	= property(lambda _: Path( _.p.project_dir, _.p.run_name))
		sweep_path_id:  str     = property(lambda _: f'{_.entity}/{_.project}/{_.exp_name}')
		wb_type: 		str		= property(lambda _: _.wb_sweep*'sweeps' or _.run_sweep*f'groups' or 'runs')
		run_url: 		str		= property(lambda _: f'www.wandb.ai/{_.entity}/{_.project}/{_.wb_type}/{_.exp_name}')
  

	class distribute(Sub):
		head:			bool	= True 
		dist_mode: 		str		= 'pyfig'  # options: accelerate
		dist_id:		str		= ''
		sync:			list	= ['grads',]
		accumulate_step:int		= 5

		gpu_id_cmd:		str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'
		gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_.gpu_id_cmd)).split('.')[0])
		dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])

	class resource(Sub):
		option: 		type	= niflheim_resource
		submit: 		bool	= False
		
		submit_cluster: Callable= None
		script:			Callable= None

	class git(Sub):
		branch:     str     = 'main'
		remote:     str     = 'origin' 
  
		commit_id_cmd:	str 	= 'git log --pretty=format:%h -n 1'
		commit_id:   	list	= property(lambda _: run_cmds(_.commit_id_cmd, cwd=_.project_dir))
		commit_cmd:		str 	= 'git commit -a -m "run"' # !NB no spaces in msg 
		commit: 		list	= property(lambda _: run_cmds(_.commit_cmd, cwd=_.project_dir)) 
		pull_cmd:		str 	= ['git fetch --all', 'git reset --hard origin/main']
		pull:   		list	= property(lambda _: run_cmds(_.pull_cmd, cwd=_.project_dir))
  
  
	home:				Path	= Path().home()
	dump:               Path    = property(lambda _: Path('dump') )
	dump_exp_dir: 		Path 	= property(lambda _: _.dump/'exp')
	tmp_dir:            Path	= property(lambda _: _.dump/'tmp')
	project_dir:        Path    = property(lambda _: _.home / 'projects' / _.project)
	cluster_dir: 	Path    	= property(lambda _: Path(_.exp_dir, 'cluster'))
	exchange_dir: 	Path    	= property(lambda _: Path(_.exp_dir, 'exchange'))

	debug: bool    = False
	env_log_path = 'dump/tmp/env.log'
	d_log_path = 'dump/tmp/d.log'

	def __init__(ii, notebook:bool=False, sweep: dict=None, **init_arg):
	
		ii.setup_sub_cls()

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

				is_first_run = bool(i)

				group_exp = group_exp or (is_first_run and ii.sweep.run_sweep)
				ii.setup_exp_dir(group_exp= group_exp, force_new_id= True)
				
				base_d = cls_to_dict(ii, sub_cls=True, flat=True, ignore=['sweep',])
				run_d = base_d | run_d

				if is_first_run:
					ii.debug_log([dict(os.environ.items()), run_d], [ii.env_log_path, ii.d_log_path])
				
				ii.resource.submit_cluster(run_d)

			sys.exit(ii.wb.run_url)

	def to_torch(ii, device, dtype):
		import torch
		d = cls_to_dict(ii, sub_cls=True, flat=True, ignore=['sweep',])
		d = {k:v for k,v in d.items() if isinstance(v, (np.ndarray, np.generic, list))}
		d = {k:torch.tensor(v, dtype=dtype, device=device, requires_grad=False) for k,v in d.items() if not isinstance(v[0], str)}
		ii.debug_log([d,], [ii.tmp_dir/'to_torch.log'])
		ii.update_configuration(d)



""" 
# Pyfig Docs
## Prerequisite
- wandb api key 
- change secrets file

## Usage 10/1/23
- python run.py

## arguments:
- --submit (submit to Niflheim, UPDATE SO USER INDEPENDENT IN USER.PY)
- --n_gpu x (sets for x gpus)
- --debug (creates log files in dump/tmp)
- --run_sweep (runs sweep from variables in sweep subclass parameters dict)
- group_exp:	force single runs into same folder for neatness 
- init_arg: 
	can be kw arguments n_b=126 
	or dictionaries model=dict(n_layer=2, n_hidden=10)
	or a sweep configuration sweep=dict(parameters=...)

## Examples:
- python run.py --submit --run_sweep --debug --n_gpu 8
- python run.py --submit --run_sweep

## Issues 
- sub classes can NOT call each other
- properties can NOT recursively call each other
- no dictionaries (other than sweep) as configuration args
- if wandb fails try # settings  = wandb.Settings(start_method='fork'), # idk y this is issue, don't change

### Accumulate
potential issues:
- loading / unloading too fast / slow? Crashes occasionally.
		
# Wandb Docs
- entity is the team name


# Useful cmds
_kill_all = 'ssh amawi@svol.fysik.dtu.dk "killall -9 -u amawi"'
_kill_all_cmd = 'ssh user@server "killall -9 -u user"'
_cancel_job_cmd = f'scancel {cluster.job_id}'

"""