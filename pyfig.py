from pathlib import Path
import numpy as np

from pyfig_utils import PyfigBase, Sub, niflheim_resource

from dump.systems import systems
from dump.user_secret import user

class Pyfig(PyfigBase):

	user: 				str 	= user
	project:            str     = 'hwat'
	run_name:       	Path	= 'run.py'
 
	exp_name:       	str		= 'demo'
	profile: 			bool 	= True
	
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
		opt_name: 		str		= 'RAdam'
		lr:  			float 	= 0.001
		betas:			list	= [0.9, 0.999]
		eps: 			float 	= 1e-4
		weight_decay: 	float 	= 0.0
		hessian_power: 	float 	= 1.0

	class sweep(PyfigBase.sweep):
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

	def __init__(ii, notebook:bool=False, sweep: dict=None, init_arg: dict=None, **other_arg) -> None:

		super().__init__(notebook=notebook, init_arg=init_arg, **other_arg)

		system = systems.get(ii.data.system, {})

		super().__post_init__(**system)
  
		ii.runfig() # docs:runfig
  
  
		"""  
		# pyfig
		## pyfig:todo
  
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

		## pyfig:prereq
		- wandb api key 
		- change secrets file

		# docs
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