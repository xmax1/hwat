from pathlib import Path
import numpy as np
from datetime import datetime

from typing import Callable, Any

from things.pyfig_utils import PyfigBase, Param, PlugIn
from things.resource_utils import niflheim
from things.distribute_utils import naive, hf_accel

from dump.systems import systems
from dump.user_secret import user

import wandb

class Pyfig(PyfigBase):

	user: 				str 	= user
 
	project:            str     = 'hwat'
	run_name:       	Path	= 'run.py'
	load_exp_state:		str		= ''

	exp_name:       	str		= '' # default is demo
	exp_id: 			str		= ''
	group_exp: 			bool	= False

	mode: 				str		= ''
	multimode: 			str		= 'train:eval'

	debug: 				bool    = False
	run_sweep:      	bool    = False
	
	seed:           	int   	= 808017424 # grr
	dtype:          	str   	= None # torch.float32  # keep torch out ofthe namespace for now
	cudnn_benchmark: 	bool 	= False

	n_step:         	int   	= 1000
	n_pre_step:    		int   	= 250
	n_eval_step:        int   	= 100
	n_total_step: 	 	int   	= property(lambda _: _.n_step + _.n_pre_step + _.n_eval_step)

	step: 				int 	= None
	log_metric_step: 	int   	= property(lambda _: 2 if _.debug else _.n_step//20)
	log_state_step: 	int   	= property(lambda _: 2 if _.debug else _.n_step//10)
 
	opt_obj_key:			str		= 'e'
	opt_obj_op: Callable = property(lambda _: lambda x: x.std())

	class data(PlugIn):
		n_b: int = 64
	
	class app(PlugIn):

		system_name: str		= ''
		system_id = property(lambda _: [[int(a_z_i), a_i.tolist()] for a_z_i, a_i in zip(_.a_z, _.a)])

		charge:     int         = 0
		spin:       int         = 0
		a:          np.ndarray  = np.array([[0.0, 0.0, 0.0],])
		a_z:        np.ndarray  = np.array([16,])

		n_corr:     int         = 20
		n_equil_step:int		= 0
		acc_target: int         = 0.5

		hf: 	   Any          = property(lambda _: _._hf)
		mol: 	   Any	        = property(lambda _: _._mol)

		n_e:        int         = property(lambda _: int(sum(_.a_z)))
		n_u:        int         = property(lambda _: (_.spin + _.n_e)//2)
		n_d:        int         = property(lambda _: _.n_e - _.n_u)
		
		n_equil_step:int        = property(lambda _: 1000//_.n_corr)

		def __init__(ii, parent=None) -> None:

			super().__init__(parent=parent)
			# 	_system_id = property(lambda _: 
			# 	'_'.join([
			# 		f'{a_z_i:.0f}-{a_i[0]:.2f}-{a_i[1]:.2f}-{a_i[2]:.2f}' 
			# 	for a_z_i, a_i in zip(_.a_z, _.a)
			# ]))
			# ii.ignore = ['hf', 'mol']
			ii.system_name = ii.system_name

			from pyscf import gto

			mol: gto.Mole = gto.Mole(
				atom=ii.system_id, basis='sto3g', unit='bohr'
			)

			# print(ii._system_id.replace('-', ' ').replace('_', ';'))
			print(ii.system_id)
			mol.spin = ii.spin
			mol.charge = ii.charge
			mol.build()
			hf = mol.UHF()
			hf.kernel()

			ii._hf = hf
			ii._mol = mol


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
		n_pv:           int     = 32
		n_fb:           int     = 3
		n_det:          int     = 4
		
		n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)

	class opt(PyfigBase.opt):
		_available_opt: list 	= ['AdaHessian', 'RAdam']
		opt_name: 		str		= 'RAdam'
		lr:  			float 	= 0.01
		betas:			list	= [0.9, 0.999]
		eps: 			float 	= 1e-4
		weight_decay: 	float 	= 0.0
		hessian_power: 	float 	= 1.0

		class scheduler(PlugIn):
			_prefix: 	str 	= 'sch_'
			n_scheduler_step: 	int   	= property(lambda _: _._p._p.n_step + _._p._p.n_pre_step)

			sch_default:str 	='OneCycleLR'

			sch_name: 	str		= 'OneCycleLR'
			sch_max_lr:	float 	= 0.01
			sch_epochs: int 	= 1

	class sweep(PyfigBase.sweep):
		sweep_method: 	str		= 'grid'
		sweep_name: 	str		= 'study'	

		parameters: 	dict 	= dict(
			# dtype			=	Param(values=[torch.float32, torch.float64], dtype=str), # !! will not work
			# n_b			=	Param(values=[512, 1024, 2048], dtype=int), # 

			opt_name		=	Param(values=['AdaHessian',  'RAdam'], dtype=str),
			lr				=	Param(domain=(0.0001, 1.), log=True),
			sch_max_lr		=	Param(values=[0.1, 0.01, 0.001], dtype=float),
			weight_decay	= 	Param(domain=[0.0, 1.], dtype=float, condition=['AdaHessian',]),
			hessian_power	= 	Param(values=[0.5, 1.], dtype=float, condition=['AdaHessian',]),

			# n_sv	= 	Param(values=[16, 32, 64], dtype=int),
			# n_pv	= 	Param(values=[16, 32, 64], dtype=int),
			# n_det	= 	Param(values=[1, 2, 4, 8], dtype=int),
			# n_fb	= 	Param(values=[1, 2, 3, 4], dtype=int),
			# n_b	= 	Param(values=[512, 1024, 2048, 4096], dtype=int),
		)

	class dist(naive):
		dist_name = 'naive'  # options: 'naive', 'hf_accel'

	class resource(niflheim):
		env: 			str     = 'zen'
		n_gpu: 			int 	= 1

	class wb(PyfigBase.wb):
		wb_mode = 'online'

	class tag(PlugIn):
		pre: str = 'pre'
		train: str = 'train'
		eval: str = 'eval'
		record: str = 'record'

	zweep: str = ''

	def __init__(ii, notebook: bool=False, sweep: dict={}, c_init: dict={}, **other_arg) -> None:

		import sys
		print('--_debug' in sys.argv, sys.argv)
		if '--_debug' in sys.argv:
			c_init = dict(
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

		### under construction
		# import sys
		# print(sys.argv[0][1:])
		# if any(['init_accelerate' in k for k in sys.argv[0][1:].split(' ')]):
		# 	print('loading accelerate')
		# 	from accelerate import accel
		# 	accel = accel()
		# 	c_init.update(dict(accel=accel))
		### under construction

		print('\npyfig:init')
		super().__init__(notebook=notebook, c_init=c_init, sweep=sweep, **other_arg)

		ii.update(systems.get(ii.app.system_name, {}))

		ii.run_local_or_submit()

		"""
		1- Fix the batch size for O2_neutral by mem_maxing 5-25 on base model

		2- Base model
		base = dict(
			n_sv=32,
			n_pv=32,
			n_fb=3,
			n_det=4,
		)

		3- 
		"""

		""" New PlugIns 
		- aim https://github.com/aimhubio/aim
		- slurm exp management https://github.com/TUM-DAML/seml
		- """

		""" todo
		# now
		- clean pyfig 
		- accel finish with split dataloader

		# soon
		- export model to latex math
		- ignore issue, how to consistently ignore an attr


		# later
		- autogen a configurable demo graph of code, which looks at the model
		- delay import of ml framework until function calls
		"""

		""" run list
		- 20 gpus
		- pretraining

		
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
			dist_name		= 	Param(values=['naive', 'hf_accel'], dtype=str), 
			
			
		)

		python run.py --submit --time 00:05:00 --n_b 32 --n_gpu 1 --mode train --dist_name hf_accel



		python run.py --submit --mode opt_hypam

		python run.py --time 01:00:00 --submit --dist_name hf_accel \
		--mode max_mem --system O2_neutral_triplet --exp_name show~max_mem_sweep \
		--n_pre_step 50 --n_step 200 --n_gpu 2
		
		--zweep n_gpu-1-2-4-8-10-int


		### dummy everything
		python run.py --time 01:00:00 --submit --multimode train:eval:max_mem:opt_hypam \
		--exp_name ~debug --n_step 40 --n_pre_step 20 --dist naive --n_gpu 2 --n_b 128 --a_z [4]

		### dummy accelerate
		python run.py --time 00:10:00 --submit --dist_name hf_accel \
		--mode train --system O2_neutral_triplet --exp_name ~debug \
		--n_pre_step 10 --n_step 100 --n_gpu 2

		### dummy dist
		python run.py --time 00:10:00 --submit --dist_name naive \
		--mode train --exp_name ~debug \
		--n_pre_step 100 --n_step 1000 --n_gpu 2 --a_z [4]

		### real accelerate 
		python run.py --time 01:00:00 --submit --dist_name hf_accel \
		--mode train --system O2_neutral_triplet --exp_name show~n_gpu_sweep \
		--n_pre_step 1000 --n_step 2000 --zweep n_gpu-1-2-4-8-int

		python run.py --time 02:00:00 --submit --dist naive \
		--multimode train-record:eval --system O2_neutral_triplet --exp_name share~n_gpu \
		--n_pre_step 1000 --n_step 10000 --zweep n_gpu-1-2-4-8-int
				
		### dummy opt
		python run.py --time 00:05:00 --submit --dist_name naive \
		--mode opt_hypam --a_z [4] --exp_name ~debug \
		--n_pre_step 50 --n_step 100 --n_gpu 1 --n_trials 10

		### real opt
		python run.py --time 04:00:00 --submit --system O2_neutral_triplet --dist_name naive \
		--mode opt_hypam --exp_name show~opt \
		--n_pre_step 100 --n_step 500 --n_gpu 1 --n_trials 1000

		
		### big
		python run.py --time 02:00:00 --submit --dist_name hf_accel \
		--mode train --exp_name show~50e \
		--n_pre_step 1000 --n_step 10000 --n_gpu 8 --a_z [50]


		"""

		# 
		# )

		# python run.py --time 03:00:00 --submit --multimode max_mem:opt_hypam --exp_name sweep~memopt --zweep a_z-16-17
		# python run.py --time 03:00:00 --submit --multimode max_mem:opt_hypam --exp_name sweep~memopt --zweep "[[16],[17]]"
		# python run.py --time 03:00:00 --submit --multimode max_mem:opt_hypam --exp_name sweep~memopt --zweep a_z-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30


		# python run.py --submit --mode train --n_det 1 --n_step 10000 --a_z [4] --exp_name stab~4 --n_pre_step 1000
		# python run.py --submit --mode train --n_det 1 --n_step 10000 --a_z [16] --exp_name stab~16 --n_pre_step 1000
		# python run.py --submit --dist hf_accel --n_gpu 2 --exp_name demo~opt_hypam --mode opt_hypam --time 12:00:00 --system O2_neutral_triplet
		# a_z		= 	Param(values=[[i,] for i in range(5, 50)], dtype=int),
		# get_mem_max 15

		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 01:00:00 --mode --max_mem

		# 26 / 1 / 23
		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 00:20:00 --dist naive
		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 00:20:00 --dist naive --cudnn_benchmark
		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 00:20:00 --dist hf_accel
		# python run.py --submit --a_z [16] --exp_name speed1~cudnn-dist --time 00:20:00 --dist hf_accel --cudnn_benchmark

		# python run.py --submit --cudnn_benchmark --exp_name opt_hypam1~O2 --mode opt_hypam-record --time 12:00:00 --n_step 500 --n_b 512

		# python run.py --submit --a_z [16] --dist naive --cudnn_benchmark --exp_name sweep-n_b --mode max_mem --time 12:00:00
		# python run.py --submit --dist naive --cudnn_benchmark --exp_name sweep-a_z --group_exp --time 00:10:00
		# python run.py --submit --exp_name dist --group_exp --time 01:00:00 --a_z [30] --dist naive --mode max_mem --n_gpu 10
		# python run.py --submit --exp_name dist --group_exp --time 01:00:00 --a_z [30] --dist hf_accel --mode max_mem --n_gpu 10
		# for a_z in [[i,] for i in range(10, 60, 2)]:
		# 	run_d = dict(a_z=a_z)

	def record_app(ii, opt_obj_all: list):

		atomic_id = "-".join([str(int(float(i))) for i in ii.app.a_z.flatten()])
		spin_and_charge = f'{ii.app.charge}_{ii.app.spin}'
		geometric_hash = f'{ii.app.a.mean():.0f}'
		exp_metaid = '_'.join([atomic_id, spin_and_charge, geometric_hash])

		columns = ["charge_spin_az0-az1-..._pmu", "Energy", "Error (+/- std)"]
		data = [exp_metaid, np.array(opt_obj_all).mean(), np.array(opt_obj_all).std()]

		print('post_process:data = \n', data)

		Result = wandb.Table(columns=columns)  # , data=data
		Result.add_data(*data)
		# wandb.Table(dataframe=my_df)
		wandb.log(dict(Result=Result))

		return True

		# api = wandb.Api()
		# run = api.run(c.wb.wb_run_path)
		# c: dict = run.config
		# history = run.scan_history(keys=['e',])
		# opt_obj = np.asarray([row['e'] for row in history])

"""  
# pyfig
## pyfig:todo
### docs:pyfig:load

- baseline cmd with pretty table
- copy all code to run dir
- generalisation refactor 

- https://jvmc.readthedocs.io/en/latest/index.html

- normalised pretraining
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
--zero_stage (int) — DeepSpeed’s ZeRO optimization stage.
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
--fsdp_min_num_params (int) — FSDP’s minimum number of parameters for Default Auto Wrapping.
--fsdp_sharding_strategy (int) — FSDP’s Sharding Strategy.
--fsdp_auto_wrap_policy (str) — FSDP’s auto wrap policy.
--fsdp_transformer_layer_cls_to_wrap (str) — Transformer layer class name (case-sensitive) to wrap, e.g, BertLayer, GPTJBlock, T5Block …
--fsdp_backward_prefetch_policy (str) — FSDP’s backward prefetch policy.
--fsdp_state_dict_type (str) — FSDP’s state dict type.

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