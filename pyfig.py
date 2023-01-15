from pathlib import Path
import numpy as np

from pyfig_utils import PyfigBase, Sub, niflheim_resource

from dump.user_secret import user

class Pyfig(PyfigBase):

	user: 				str 	= user
	project:            str     = 'hwat'
	run_name:       	Path	= 'run.py'
 
	exp_name:       	str		= 'demo'
	group_exp: 			bool	= False
	
	seed:           	int   	= 808017424 # grr
	dtype:          	str   	= 'float32'
	n_step:         	int   	= 10000
	log_metric_step:	int   	= 10
	log_state_step: 	int   	= 10
	
	class data(PyfigBase.data):
		system: 	str			= '' # overwrites base

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

	class model(PyfigBase.model):
		compile_ts: 	bool	= False
		compile_func:	bool	= False
		optimise_ts:	bool	= False
		optimise_aot:	bool 	= False
		with_sign:      bool    = False
		functional: 	bool	= True

		terms_s_emb:    list    = ['ra', 'ra_len']
		terms_p_emb:    list    = ['rr', 'rr_len']
		ke_method:      str     = 'grad_grad'
		n_sv:           int     = 32
		n_pv:           int     = 16
		n_fb:           int     = 2
		n_det:          int     = 1
  
		n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)
  
	class opt(PyfigBase.opt):
		lr: 			float	= 0.001
		name: 			str		= 'Adam'

	class sweep(Sub):
		run_sweep: 		bool	= False
		method: 		str		= 'grid'
		parameters: 	dict 	= 	dict(
			n_b  = dict(values=[16, 32, 64]),
		)

	class distribute(PyfigBase.distribute):
		dist_mode: 		str		= 'pyfig'  # options: accelerate
		sync_step:		int		= 5

	class resource(niflheim_resource):
		submit:    		bool	= False
		env: 			str     = 'lumi'
		n_gpu: 			int 	= 1

	class wb(PyfigBase.wb):
		wb_mode = 'online'
		print('loading wandb from base class')

	def __init__(ii, notebook:bool=False, sweep: dict=None, **init_arg) -> None:

		super().__init__(notebook=notebook, sweep=sweep, **init_arg)

		if ii.data.system:
			from dump.systems import systems
			system = systems.get(ii.data.system, None)
			if system is None:
				exit(ii.data.system + ' not in systems dict (located dump/systems)')
			ii.pr(system)
			print('Loading system, currently overwritten by sys_arg so be careful not to double set')
			ii.update_configuration(system, sweep=ii.sweep.d if ii.sweep.run_sweep else None)

		ii.runfig() # docs:runfig
  
		"""  # Pyfig Docs
		- todo
		- try to compile the sampler and see where fails

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

		# example
		- python run.py --submit --run_sweep --debug --n_gpu 8
		- python run.py --submit --run_sweep

		# issues 

		## issue:sub_classes
		- sub classes can NOT call each other
		- properties can NOT recursively call each other
		- no dictionaries (other than sweep) as configuration args
		- if wandb fails try # settings  = wandb.Settings(start_method='fork'), # idk y this is issue, don't change

		# docs
		## docs:runfig
		- prereq for 'runfig' complete transform
		- needed to be this way round to ensure the system is initialised before running
		- didn't put systems in base config bc want to generalise to other projects

		## docs:accumulate
		potential issues:
		- loading / unloading too fast / slow? Crashes occasionally.
				
		## docs:wandb
		- entity is the team name

		## docs:torchscript
		- unsupported
			- https://pytorch.org/docs/stable/jit_unsupported.html 
			- Functions which construct tensors from non-tensor inputs do not support the requires_grad argument, except for torch.tensor. (ie torch.ones)

		## docs:profiler
		1- --> wandb --> Artifacts --> files --> trace
		https://wandb.ai/wandb/trace/reports/Using-the-PyTorch-Profiler-with-W-B--Vmlldzo5MDE3NjU
		2- tensorboard --logdir=c.profile_dir
		browser: http://localhost:6006/pytorch_profiler
		https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
		
		### docs:compile-torchscript-model
		# - Final[torch.Tensor] not valid type
		# - register_buffer way to include tensor constants

		## docs:useful_cmd
		_kill_all = 'ssh amawi@svol.fysik.dtu.dk "killall -9 -u amawi"'
		_kill_all_cmd = 'ssh user@server "killall -9 -u user"'
		_cancel_job_cmd = f'scancel {cluster.job_id}'

		
		"""