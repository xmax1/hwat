from pathlib import Path
import numpy as np

from pyfig_utils import PyfigBase, Param, niflheim_resource, Sub, dict_to_cmd

from dump.systems import systems
from dump.user_secret import user
from datetime import datetime

class Pyfig(PyfigBase):

	user: 				str 	= user
 
	project:            str     = 'hwat'
	run_name:       	Path	= 'run.py'
	load_exp_state:		str		= ''

	exp_name:       	str		= '' # default is demo
	exp_id: 			str		= ''
	group_exp: 			bool	= False

	multimode: 			str		= 'train:evaluate' # 'max_mem:profile:opt_hypam:train:evaluate'
	mode: 				str		= ''
	debug: 				bool    = False
	run_sweep:      	bool    = False
	
	seed:           	int   	= 808017424 # grr
	dtype:          	str   	= 'float32'

	n_step:         	int   	= 100
	n_eval_step:        int   	= 100
	n_pre_step:    		int   	= 50
	step: 				int 	= None

	log_metric_step:	int   	= 10
	log_state_step: 	int   	= 10
 
	save: bool = False
	load_state: bool = False
	
	class data(PyfigBase.data):
		system: 	str			= '' # overwrites base

		charge:     int         = 0
		spin:       int         = 0
		a:          np.ndarray  = np.array([[0.0, 0.0, 0.0],])
		a_z:        np.ndarray  = np.array([4.,])

		n_b:        int         = 512
		n_corr:     int         = 20
		acc_target: int         = 0.5

		n_equil_step:	int		= 100
		# n_equil_step:int        = property(lambda _: 1e6//_.n_corr)
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
		opt_name: 		str		= 'RAdam'
		scheduler_name: str		= 'OneCycleLR'
		lr:  			float 	= 0.0001
		max_lr:			float 	= 0.01
		betas:			list	= [0.9, 0.999]
		eps: 			float 	= 1e-4
		weight_decay: 	float 	= 0.0
		hessian_power: 	float 	= 1.0

	class sweep(PyfigBase.sweep):
		method: 		str		= 'grid'
		parameters: 	dict 	= dict(
			lr= Param(domain=(0.01, 0.0001), log=True)
		)

	class distribute(PyfigBase.distribute):
		dist_method = 'pyfig'

	class resource(niflheim_resource):
		env: 			str     = 'zen'
		n_gpu: 			int 	= 1

	class wb(PyfigBase.wb):
		wb_mode = 'online'
		print('loading wandb from base class')

	def __init__(ii, notebook: bool=False, sweep: dict={}, c_init: dict={}, **other_arg) -> None:

		print('initialising')
		super().__init__(notebook=notebook, c_init=c_init, sweep=sweep, **other_arg)

		system = systems.get(ii.data.system, {})

		print('initialising system')
		# super().__post_init__(**system)
		ii.update(system)
  
		ii.pf_submit() # docs:runfig

		"""  
		# pyfig
		## pyfig:todo
		### docs:pyfig:load
		- load exp_dir
		- load sub_cls
		- if c.load_exp is True

		- normalised pretraining

		- save eval r and stats compressed

		- estimated time until finished from # electrons, batch size 
		- accelerate test
		- accelerate seeds test 
		- buy food 
		- call rents
		- cook dinner
		- buy cigarettes 
		- look at week
		- wandb url goes to group :X:
		- groups are exp_id + mode + reinit# :X:
		- wandb reports
		- distributed metrics: e_std_across_nodes
		- save best mem for c in dump 
  
		- # size_param: 	list	= property(lambda _: [datetime.now().strftime("%d-%m-%y:%H-%M-%S"), _.n_b, _.n_e, _.n_fb, _.n_sv, _.n_pv])
		- for memory map


		- save opt c
		- optimise so works well for 10 electrons
		- name exps "optimising c"
		- save as cmd line for line
		- save as .c file 

		- load c (but only model and data)
		if c.load.load_exp_dir:

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
		- sub classes can NOT call each other
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
		## docs:distribute:accelerator
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