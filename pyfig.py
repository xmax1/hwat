
import sys
from pathlib import Path
import numpy as np

from typing import Callable, Any

from things.pyfig_utils import PyfigBase 
from things.utils import PlugIn

from dump.systems import systems
from dump.user_secret import user

"""
base classes contain the following:
- globals (variables that are used across the entire project)"""


""" 
# issues
* ** *** **** ***** importance
- create single gpu base class 
- create hidden plugin
- convert wb to general 


- type_me argument parsing fails for empty lists. Error is caught but unsure if other effects so far. 
- paths and dirs need to be strings, not Nones
- exp_name something/something~something is not parsed correctly
- on niflheim, use export TMPDIR=/tmp for the node local scratch space.
- on niflheim, special scratch space /home/scratch2/$USER/ across nodes
- / not allowed in exp_name
- fail flag does not work
- all dirs created when not needed
- generalised to numpy and torch tensors conversions with None and other catches ****
- informed syntax for flat calls ***
- sync forced by run loop function *
- naive distribution doesn't synchronize the opt hypam correct
- run_local_or_submit --> if: run_submit statement. Follow 'external condition' or 'this_is_noop' pattern
- group exp method clean -> create_new_exp_dir is property
- float64 torch
- include in opt_hypam *
- distribution schema 
	- include docs in base classes
	- distribute / resource
		- tasks -> threads -> processes -> nodes -> gpus -> cpus -> rank... 
- fix the memory error catching in optuna, now it hangs
- hostname IS NOT THE NODENAME S007 NEQ S008

- DISTRIBUTION OF HF_ACCEL 
- 1. write a hostfile (plugin)
Distributed GPUs:
  Arguments related to distributed GPU training.

  --gpu_ids GPU_IDS     What GPUs (by id) should be used for training on this machine as a comma-seperated list
  --same_network        Whether all machines used for multinode training exist on the same local network.
  --machine_rank MACHINE_RANK
                        The rank of the machine on which this script is launched.
  --main_process_ip MAIN_PROCESS_IP
                        The IP address of the machine of rank 0.
  --main_process_port MAIN_PROCESS_PORT
                        The port to use to communicate with the machine of rank 0.
  --rdzv_conf RDZV_CONF
                        Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).
  --max_restarts MAX_RESTARTS
                        Maximum number of worker group restarts before failing.
  --monitor_interval MONITOR_INTERVAL
                        Interval, in seconds, to monitor the state of workers.
- -2 to switch off sync_step / logging etc 

- pyfig 'create_new' needed to pass a deepcopy object to a run mode ** 



# Create the summary run.
# summary = wandb.init(project="optuna",
# 					name="summary",
# 					job_type="logging")

# Getting the study trials.
# trials = study.trials

# WandB summary.
# for step, trial in enumerate(trials):
# 	# Logging the loss.
# 	summary.log({"mse": trial.value}, step=step)

# 	# Logging the parameters.
# 	for k, v in trial.params.items():
# 		summary.log({k: v}, step=step)


# distribution
- local rank: rank on a node
- global rank: rank across nodes
- group rank: rank within a group (ie node)
You can think of world as a group containing all the processes for your distributed training.

# patterns
- use of 'external condition' or 'this_is_noop' pattern
- main method and shortcut
	- group_exp vs ~
	- run_sweep vs zweep 

# useful 
- variable descriptions things/pyfig_def.py

# requirements
## distribution
- each gpu must have a unique rank env variable

# very basic test
- python run.py

# other test
python run.py --submit --n_gpu 2 --n_b 2 --mode train --_debug --plugin_name naive --exp_name test

# debug 1 
python run.py --submit --time 01:00:00 --n_gpu 2 --debug --multimode pre:opt_hypam:pre:train:eval \
--log_metric_keys ['all'] --n_pre_step 20 --n_opt_hypam_step 20 --n_trials 4 --n_train_step 20 --n_eval_step 20

# debug 2
python run.py --submit --time 01:00:00 --n_gpu 2 --multimode pre:opt_hypam:pre:train:eval \
--log_metric_keys ['all'] 

# run 
python run.py --submit --time 01:00:00 --n_gpu 2 --multimode pre:opt_hypam:pre:train:eval --log_metric_keys ['all'] \
--system_name O2_neutral_triplet --exp_name ~ScaleO2_v3

python run.py --submit --time 01:00:00 --multimode pre:opt_hypam:pre:train:eval --log_metric_keys ['all'] \
--system_name O2_neutral_triplet --exp_name ~ScaleO2_v2 --zweep n_gpu-1-2-4-8-10-int --n_b 128

python run.py --submit --time 01:00:00 --multimode pre:opt_hypam:pre:train:eval --log_metric_keys ['all'] \
--system_name O2_neutral_triplet --exp_name ~ScaleO2_v2 --n_gpu 20 --n_b 128

scancel 5942494 \
5943347 \
5943348 \
5943349 \
5943350 \
5943345 \
5943346 \
 \
 \
 \


"""

from things.dist_repo import DistBase, Naive, HFAccelerate, SingleProcess
from things.logger_repo import LoggerBase, Wandb
from things.resource_repo import ResourceBase, Niflheim
from things.sweep_repo import SweepBase, Optuna, Param
from things.gen_repo import OptBase, DataBase, SchedulerBase, PathsBase, ModelBase

class Pyfig(PyfigBase):

	user: 				str 	= user
	project:            str     = 'hwat'
	env: 				str     = 'zen'
 
	run_name:       	Path	= 'run.py'

	exp_name:       	str		= '' # default is demo
	exp_id: 			str		= ''
	group_exp: 			bool	= False
	lo_ve_path: 		str 	= '' # for LOad & saVE -> lo_ve

	mode: 				str		= '' # one or the other
	multimode: 			str		= 'pre:train:eval'

	debug: 				bool    = False
	
	seed:           	int   	= 808017424 # grr
	dtype:          	str   	= None # torch.float32  # keep torch out ofthe namespace for now
	cudnn_benchmark: 	bool 	= True

	n_log_metric:		int  	= 50
	n_log_state:		int  	= 1

	@property
	def is_logging_process(ii: PyfigBase):
		return ii.mode==ii.opt_hypam_tag or ii.dist.head
	# is_logging_process: bool 	= property(def(ii: PyfigBase): ; return ii.mode==ii.opt_hypam_tag or ii.dist.head)

	log_state_keys:     list 	= [] # ['opt_obj', 'r', 'grads', 'params', ...] 'all' is all
	log_metric_keys: 	str		= [] # ['opt_obj', 't_per_it', 'max_mem_alloc']
 
	opt_obj_key:			str		= 'e'
	opt_obj_op: Callable = property(lambda _: lambda x: x.std())
	
	class data(DataBase):
		n_b: int = 1024
		loader_n_b: int = 1
	
	class app(PlugIn):
		
		system_name: str		= ''
		system_id = property(lambda _: [[int(a_z_i), a_i.tolist()] for a_z_i, a_i in zip(_.a_z, _.a)])
		system_id_path: str = property(lambda _: _.p.paths.dump_dir / f'{_.system_id}.txt')

		charge:     int         = 0
		spin:       int         = 0
		a:          np.ndarray  = np.array([[0.0, 0.0, 0.0],])
		a_z:        np.ndarray  = np.array([4,])

		n_corr:     int         = 10
		n_equil_step:int		= 0
		acc_target: int         = 0.5
		init_data_scale: float  = 5.

		mo_coef:   np.ndarray  = None
		_hf 	= None
		_mol 	= None
		mol: 	   Any	        = property(lambda _: _._mol)

		n_e:        int         = property(lambda _: int(sum(_.a_z)))
		n_u:        int         = property(lambda _: (_.spin + _.n_e)//2)
		n_d:        int         = property(lambda _: _.n_e - _.n_u)
		
		n_equil_step:int        = property(lambda _: 1000//_.n_corr)

		# modes of operation
		loss: str        = ''  # orb_mse, vmc
		compute_energy: bool = False  # true by default

		def init_app(ii):
			""" 
			- want this to happen after pyfig is updated
			"""
			from pyscf import gto

			mol: gto.Mole = gto.Mole(
				atom=ii.system_id, basis='sto3g', unit='bohr'
			)

			print('\npyfig:pyscf: ', ii.system_id, [type(v) for v in ii.system_id], sep='\n')
			mol.spin = ii.spin
			mol.charge = ii.charge
			mol.build()
			hf = mol.UHF()
			hf.kernel()

			ii._hf = hf
			ii._mol = mol
			ii.mo_coef = np.array(hf.mo_coeff)

			ii.p.if_debug_print_d({'hf': hf, 'mol': mol, 'mo_coef': ii.mo_coef})

		def record_summary(ii, summary: dict=None, opt_obj_all: list=None) -> None:
			import wandb

			columns = ["charge_spin_az0-az1-...pmu", "Energy", "Error (+/- std)"]
			atomic_id = "-".join([str(int(float(i))) for i in ii.a_z.flatten()])
			spin_and_charge = f'{ii.charge}_{ii.spin}'
			geometric_hash = f'{ii.a.mean():.0f}'
			exp_metaid = '_'.join([atomic_id, spin_and_charge, geometric_hash])
			
			if len(opt_obj_all)==0:
				opt_obj_all = [0.0]
				print('no opt_obj_all, setting to 0.0')
			else:
				data = [exp_metaid, np.array(opt_obj_all).mean(), np.array(opt_obj_all).std()]

			print('pyfig:app:record_summary:Result ', data, sep='\n')
			Result = wandb.Table(columns=columns)
			Result.add_data(*data)
			wandb.log(dict(Result=Result) | (summary or {}))

			return True
		
		def post_init_update(ii):
			return systems.get(ii.system_name, {})

		# pyfig:docs:log
		# n_log -1 special case never
		# n_log 1 special case log last

		mode_c: dict = dict(
			pre= dict(
				n_b 			= 512,
				n_pre_step 	= 500,
				loss 			= 'orb_mse',
				log_state_keys = ['params'],
				n_log_state	= 1, 
				n_log_metric	= 5, 
				sync_step		= 5,
			),
			train= dict(
				n_b 				= 512,
				n_train_step 		= 1000,
				log_metric_keys		=  ['all'] ,
				log_state_keys 		= ['all'],
				n_log_state			= 1,
				n_log_metric		= 20,
				loss 			 	= 'vmc', # energy computed by default,
				sync_step			= 5,
			),
			eval= dict(
				n_b 				= 512,
				n_eval_step 		= 50,
				n_log_metric 		= -1, # special case never log,
				log_metric_keys	= ['opt_obj'],
				compute_energy  	= True,
				sync_step			= -1, # special case never log,
			),
			opt_hypam= dict(
				n_b 				= 512,
				n_opt_hypam_step	= 200,
				plugin_name 		= 'naive',
				loss 			 	= 'vmc', # energy computed by default,
				sync_step 			= -1, # special case never log,
				n_log_metric 		= 50, # special case log every 1,
				n_log_state 		= -1, # special case never log,
				n_trials 			= 20,
			)
		)
		debug_c: dict = dict(
			n_default_step=  	30,
			n_pre_step= 	 	0,
			n_train_step= 	 	0,
			n_eval_step= 	 	0,
			n_opt_hypam_step=   0,
			n_trials=   		6,

			exp_name= 		 '~debug',
			time= 			 '00:10:00',
			multimode= 		 'pre:opt_hypam:pre:train:eval',

			n_b= 	 		 4,
			
			max_power=	     8,
			
			a_z= 			 [4,], # in,
			n_sv=     		 16,
			n_pv=     		 8,
			n_fb=     		 2,
			n_det=    		 2,
			
			debug=			 True,
			# n_log_metric=	 10,  # can't set things important to specific runs
			# n_log_state=     1,
		)


	class model(ModelBase):
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
		n_pv:           int     = 32
		n_fb:           int     = 3
		n_det:          int     = 4
		n_final_out:	int     = 1
		
		n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)


	class opt(PyfigBase.opt):
		available_opt:  list 	= ['AdaHessian', 'RAdam']
		opt_name: 		str		= 'RAdam'
		lr:  			float 	= 0.01
		betas:			list	= [0.9, 0.999]
		eps: 			float 	= 1e-4
		weight_decay: 	float 	= 0.0
		hessian_power: 	float 	= 1.0
		
	class scheduler(PyfigBase.scheduler):
		sch_default:str 	= 'OneCycleLR'
		sch_name: 	str		= 'OneCycleLR'
		sch_max_lr:	float 	= 0.01
		sch_epochs: int 	= 1

	class sweep(Optuna):
		sweep_name: 	str		= 'study'	
		n_trials: 		int		= 20
		parameters: 	dict 	= dict(
			# dtype			=	Param(values=[torch.float32, torch.float64], dtype=str), # !! will not work
			opt_name		=	Param(values=['AdaHessian',  'RAdam'], dtype=str),
			weight_decay	= 	Param(domain=[0.01, 1.], dtype=float, condition=['AdaHessian',]),
			hessian_power	= 	Param(values=[0.5, 0.75, 1.], dtype=float, condition=['AdaHessian',]),
			lr				=	Param(domain=(0.0001, 1.), log=True, dtype=float),
			sch_max_lr		=	Param(values=[0.1, 0.01, 0.001], dtype=float),
			n_sv			= 	Param(values=[16, 32, 64], dtype=int),
			n_pv			= 	Param(values=[16, 32], dtype=int),
			n_det			= 	Param(values=[1, 4, 8, 16], dtype=int),
			n_fb			= 	Param(values=[2, 3, 4], dtype=int),
			n_b				= 	Param(values=[512], dtype=int),  # 64000
			n_opt_hypam_step= 	Param(values=[500,], dtype=int),
		)

	class dist(Naive):
		pass

	class resource(Niflheim):
		pass

	class logger(Wandb):
		log_mode = 'online'

	def __init__(ii, notebook: bool=False, sweep: dict={}, c_init: dict={}, **other_arg) -> None:

		print('\npyfig:init')
		super().__init__(notebook=notebook, c_init=c_init, sweep=sweep, **other_arg)

		ii.update(ii.app.post_init_update())

		if ii.submit:
			ii.run_submit()


		""" New PlugIns 
		- aim https://github.com/aimhubio/aim
		- slurm exp management https://github.com/TUM-DAML/seml
		- """

		""" todo
		# now
		- clean pyfig 
		- list different ranks

		# soon
		- export model to latex math
		- ignore issue, how to consistently ignore an attr
		- everything either numpy or deep learning framework
		- c.to('numpy') puts everything as np array
		- automatically have everything as numpy arrays?
		- log keys system 
		- simple metrics collecter and logger (also generalises logger)

		# later
		- autogen a configurable demo graph of code, which looks at the model
		- delay import of ml framework until function calls
		"""

		""" run list
		- 20 gpus
		- preing

		
		"""


		""" conceptual run

		- init_exp

		- init_app

		- pyfig.start: 
			- init logger

		- model.zero_grad: 

		- compute_loss:
			+ r
			+ ke
			+ pe
			+ deltar

			* names.mode_eval
			* names.mode_train
				* names.phase_pre
					+ pre_loss
				* names.phase_main
					+ loss

		- compute gradients:
			*** names.phase_train
			
		- synchronize with distribution
			*** n_gpu > 1
		
		- update model
			*** names.phase_train

		- log
			* names.mode_train
			* names.mode_train
		
		
		
		"""

		# tag record to be able to clear anything from wandb with no record tag
		# - todo
		# - run scaling exp


		"""

		c_test = dict(
			n_sv     = 16,
			n_pv     = 8,
			n_fb     = 2,
			n_det    = 2,
			n_b 	 = 4,
			multimode = 'max_mem-record:opt_hypam-record:train-record:eval-record',  # profile
			n_step   = 50,
			n_pre_step   = 50,
			n_eval_step   = 50,
			debug	= True,

		)

		parameters = dict(
			n_gpu			=	Param(values=[1,  2], dtype=int),
			plugin_name		= 	Param(values=['naive', 'hf_accel'], dtype=str), 
			
			
		)

		python run.py --submit --time 00:05:00 --n_b 32 --n_gpu 1 --mode train --plugin_name hf_accel



		python run.py --submit --mode opt_hypam

		python run.py --time 01:00:00 --submit --plugin_name hf_accel \
		--mode max_mem --system O2_neutral_triplet --exp_name show~max_mem_sweep \
		--n_pre_step 50 --n_step 200 --n_gpu 2
		
		--zweep n_gpu-1-2-4-8-10-int


		### dummy everything
		python run.py --time 01:00:00 --submit --multimode train:eval:max_mem:opt_hypam \
		--exp_name ~debug --n_step 40 --n_pre_step 20 --dist naive --n_gpu 2 --n_b 128 --a_z [4]

		### dummy accelerate
		python run.py --time 00:10:00 --submit --plugin_name hf_accel \
		--mode train --system O2_neutral_triplet --exp_name ~debug \
		--n_pre_step 10 --n_step 100 --n_gpu 2

		### dummy dist
		python run.py --time 00:10:00 --submit --plugin_name naive \
		--mode train --exp_name ~debug \
		--n_pre_step 100 --n_step 1000 --n_gpu 2 --a_z [4]

		### real accelerate 
		python run.py --time 01:00:00 --submit --plugin_name hf_accel \
		--mode train --system O2_neutral_triplet --exp_name show~n_gpu_sweep \
		--n_pre_step 1000 --n_step 2000 --zweep n_gpu-1-2-4-8-int

		python run.py --time 02:00:00 --submit --dist naive \
		--multimode train-record:eval --system O2_neutral_triplet --exp_name share~n_gpu \
		--n_pre_step 1000 --n_step 10000 --zweep n_gpu-1-2-4-8-int
				
		### dummy opt
		python run.py --time 00:05:00 --submit --plugin_name naive \
		--mode opt_hypam --a_z [4] --exp_name ~debug \
		--n_pre_step 50 --n_step 100 --n_gpu 1 --n_trials 10

		### real opt
		python run.py --time 04:00:00 --submit --system O2_neutral_triplet --plugin_name naive \
		--mode opt_hypam --exp_name show~opt \
		--n_pre_step 100 --n_step 500 --n_gpu 1 --n_trials 1000

		
		### big
		python run.py --time 02:00:00 --submit --plugin_name hf_accel \
		--mode train --exp_name show~50e \
		--n_pre_step 1000 --n_step 10000 --n_gpu 8 --a_z [50]


		"""

"""  
# pyfig
## pyfig:todo
### docs:pyfig:load

- baseline cmd with pretty table
- copy all code to run dir
- generalisation refactor 

- https://jvmc.readthedocs.io/en/latest/index.html

- normalised preing
- save eval r and stats compressed (5 min)

- estimated time until finished from # electrons, batch size 
- wandb reports
- distributed metrics: e_std_across_nodes

- buy food 
- call rents
- cook dinner
- buy cigarettes 
- look at week
- save best mem for c in dump 

- # size_param: 	list	= property(lambda _: [datetime.now().strftime("%d-%m-%y:%H-%M-%S"), _.n_b, _.n_e, _.n_fb, _.n_sv, _.n_pv])
- for memory map

## pyfig:def
- machine rank = global relative id of machine/process ??

## pyfig:qzone
- what is love 

## pyfig:usage
- python run.py

## pyfig:args
- --submit (submit to Niflheim, UPDATE SO USER INDEPENDENT IN USER.PY)
- --n_gpu x (sets for x gpus)
- --debug (creates log files in dump/tmp)
- --run_sweep (runs sweep from variables in sweep subclass parameters dict)
- group_exp:	force single runs into same folder for neatness 
- init_arg: 
	can be kw arguments n_b=126 
	or dictionaries model=dict(n_layer=2, n_hidden=10)
	or a sweep configuration sweep=dict(parameters=...)

## pyfig:run:examples
- python run.py --submit --run_sweep --debug --n_gpu 8
- python run.py --submit --run_sweep

## pyfig:issues:sub_classes
- PlugIn classes can NOT call each other
- properties can NOT recursively call each other
- no dictionaries (other than sweep) as configuration args
- if wandb fails try # settings  = wandb.Settings(start_method='fork'), # idk y this is issue, don't change
- ! at initialisation, sub_cls cannot be used for operations because they are not initialised and therefore 
- do not have access to d (dictionary property). This means the callable filter needs to happen *first

## pyfig:prereq
- wandb api key 
- change secrets file

# docs
## docs:python
- hasattr includes class attr not just instance attr
## docs:distribute:accel
- accelerate config before running anything to configure environment
- config file/params 
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

### docs:compile_ts
# - model(r) before does nothing
# - model = torch.jit.script(model, r.clone()) !!! r.clone() required - reason unknown

# PYTORCH_JIT=0  # disable jit
# run_cmds('export PYTORCH_NVFUSER_DISABLE=fallback')
# run_cmds(['PYTORCH_NVFUSER_DISABLE_FALLBACK=1', 'export PYTORCH_NVFUSER_DISABLE_FALLBACK'], silent=False)
# @sjlee0407 The issue you are encountering 	
# is because you have allreduce_post_accumulation=False, allreduce_post_accumulation_fp16=False
# Torchscript/NVFuser currently works with the above two flags set to true. 
# Setting the above two to true will also increase performance orthogonally.


## docs:optuna
Median pruning algorithm implemented in MedianPruner
Non-pruning algorithm implemented in NopPruner
Algorithm to operate pruner with tolerance implemented in PatientPruner
Algorithm to prune specified percentile of trials implemented in PercentilePruner
Asynchronous Successive Halving algorithm implemented in SuccessiveHalvingPruner
Hyperband algorithm implemented in HyperbandPruner
Threshold pruning algorithm implemented in ThresholdPruner

For RandomSampler, MedianPruner is the best.
For TPESampler, HyperbandPruner is the best.
"""

""" 
# docs:accelerate:config
# docs:accelerate
https://github.com/huggingface/accelerate/issues/647
In the context of multi-node training, you have:
local_rank, the rank of the process on the local machine.
rank, the rank of the process in the network.
To illustrate that, let;s say you have 2 nodes (machines) with 2 GPU each, you will have a total of 4 processes (p1…p4):

## docs:accelerate:config:ex1

compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2
num_machines: 1
machine_rank: 0
use_cpu: false
same_network: true
gpu_ids: 0,1

deepspeed_config: {}
fsdp_config: {}
main_process_ip: null
main_process_port: null

mixed_precision: fp16
gradient_accumulation_steps=2

main_training_function: main  # only useful tpu

## docs:accelerate:config:ex2
command_file: null                                                                                                                                        
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
gpu_ids: 0,1
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false

## docs:accelerate:config:notes
--multi_gpu 
--mixed_precision=fp16 
--num_processes=2
NB: --nproc_per_node=NUM_GPUS_YOU_HAVE

Hardware Selection Arguments:

--cpu (bool) — Whether or not to force the training on the CPU.
--multi_gpu (bool) — Whether or not this should launch a distributed GPU training.
--mps (bool) — Whether or not this should use MPS-enabled GPU device on MacOS machines.
--tpu (bool) — Whether or not this should launch a TPU training.
Resource Selection Arguments:

The following arguments are useful for fine-tuning how available hardware should be used

--mixed_precision {no,fp16,bf16} (str) — Whether or not to use mixed precision training. Choose between FP16 and BF16 (bfloat16) training. BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.
--num_processes NUM_PROCESSES (int) — The total number of processes to be launched in parallel.
--num_machines NUM_MACHINES (int) — The total number of machines used in this training.
--num_cpu_threads_per_process NUM_CPU_THREADS_PER_PROCESS (int) — The number of CPU threads per process. Can be tuned for optimal performance.
Training Paradigm Arguments:

The following arguments are useful for selecting which training paradigm to use.

--use_deepspeed (bool) — Whether or not to use DeepSpeed for training.
--use_fsdp (bool) — Whether or not to use FullyShardedDataParallel for training.
--use_megatron_lm (bool) — Whether or not to use Megatron-LM for training.
Distributed GPU Arguments:

The following arguments are only useful when multi_gpu is passed or multi-gpu training is configured through accelerate config:

--gpu_ids (str) — What GPUs (by id) should be used for training on this machine as a comma-seperated list
--same_network (bool) — Whether all machines used for multinode training exist on the same local network.
--machine_rank MACHINE_RANK (int) — The rank of the machine on which this script is launched.
--main_process_ip MAIN_PROCESS_IP (str) — The IP address of the machine of rank 0.
--main_process_port MAIN_PROCESS_PORT (int) — The port to use to communicate with the machine of rank 0.
--rdzv_conf (str) — Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,…).
--max_restarts (int) — Maximum number of worker group restarts before failing.
--monitor_interval (float) — Interval, in seconds, to monitor the state of workers.
TPU Arguments:

The following arguments are only useful when tpu is passed or TPU training is configured through accelerate config:

--main_training_function MAIN_TRAINING_FUNCTION (str) — The name of the main function to be executed in your script.
--downcast_bf16 (bool) — Whether when using bf16 precision on TPUs if both float and double tensors are cast to bfloat16 or if double tensors remain as float32.
DeepSpeed Arguments:

The following arguments are only useful when use_deepspeed is passed or deepspeed is configured through accelerate config:

--deepspeed_config_file (str) — DeepSpeed config file.
--zero_stage (int) — DeepSpeed ZeRO optimization stage.
--offload_optimizer_device (str) — Decides where (none|cpu|nvme) to offload optimizer states.
--offload_param_device (str) — Decides where (none|cpu|nvme) to offload parameters.
--gradient_accumulation_steps (int) — No of gradient_accumulation_steps used in your training script.
--gradient_clipping (float) — Gradient clipping value used in your training script.
--zero3_init_flag (str) — Decides Whether (true|false) to enable deepspeed.zero.Init for constructing massive models. Only applicable with DeepSpeed ZeRO Stage-3.
--zero3_save_16bit_model (str) — Decides Whether (true|false) to save 16-bit model weights when using ZeRO Stage-3. Only applicable with DeepSpeed ZeRO Stage-3.
--deepspeed_hostfile (str) — DeepSpeed hostfile for configuring multi-node compute resources.
--deepspeed_exclusion_filter (str) — DeepSpeed exclusion filter string when using mutli-node setup.
--deepspeed_inclusion_filter (str) — DeepSpeed inclusion filter string when using mutli-node setup.
--deepspeed_multinode_launcher (str) — DeepSpeed multi-node launcher to use.
Fully Sharded Data Parallelism Arguments:

The following arguments are only useful when use_fdsp is passed or Fully Sharded Data Parallelism is configured through accelerate config:

--fsdp_offload_params (str) — Decides Whether (true|false) to offload parameters and gradients to CPU.
--fsdp_min_num_params (int) — FSDP minimum number of parameters for Default Auto Wrapping.
--fsdp_sharding_strategy (int) — FSDP Sharding Strategy.
--fsdp_auto_wrap_policy (str) — FSDP auto wrap policy.
--fsdp_transformer_layer_cls_to_wrap (str) — Transformer layer class name (case-sensitive) to wrap, e.g, BertLayer, GPTJBlock, T5Block …
--fsdp_backward_prefetch_policy (str) — FSDP backward prefetch policy.
--fsdp_state_dict_type (str) — FSDP state dict type.

"""

""" docs:slurm

Computer architecture
The parts of a modern computer we need to understand to apply to running jobs are listed here. (Note: This is way oversimplified and intended to give a basic overview for the purposes of understanding how to request resources from Slurm, there are a lot of resources out there to dig deeper into computer architecture.)

Board
A physical motherboard which contains one or more of each of Socket, Memory bus and PCI bus.
Socket
A physical socket on a motherboard which accepts a physical CPU part.
CPU
A physical part that is plugged into a socket.
Core
A physical CPU core, one of many possible cores, that are part of a CPU.
HyperThread
A virtual CPU thread, associated with a specific Core. This can be enabled or disabled on a system. SCG typically disabled hyperthreading.
Memory Bus
A communication bus between system memory and a Socket/CPU.
PCI Bus
A communication bus between a Socket/CPU and I/O controllers (disks, networking, graphics,...) in the server.
Slurm complicates this, however, by using the terms core and cpu interchangeably depending on the context and Slurm command. --cpus-per-taks= for example is actually specifying the number of cores per task.
"""