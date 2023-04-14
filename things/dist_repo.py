
import torch
from pathlib import Path
from time import sleep
import optree
import os
import gc
from typing import Callable, Any
import numpy as np
import torch 
import numpy as np
from torch import nn

from .core import dump, load, run_cmds, find_free_port, add_to_Path, dict_to_cmd
from .aesthetic import print_table

from .utils import PlugIn	

this_dir = Path(__file__).parent
hostname = os.environ.get('HOSTNAME', '') # if n node > 1 this is 

class DistBase(PlugIn):
	
	dist_name: 		str = 'Base'
	n_launch: 		int = 1
	n_worker:   	int = property(lambda _: _.p.resource.n_gpu)
	ready: 			bool = True
	sync_every: 	int = 1

	rank_env_name: 	str		= 'RANK'
	rank: 			int 	= property(lambda _: int(os.environ.get(_.rank_env_name, '-1')))
	head: 			bool 	= property(lambda _: _.rank==0)
	gpu_id: 		str		= property(lambda _: ''.join(run_cmds(_._gpu_id_cmd, silent=True)).split('.')[0])
	dist_id: 		str 	= property(lambda _: _.gpu_id + '-' + hostname.split('.')[0])
	pid: 			int = property(lambda _: _.rank)

	_gpu_id_cmd:	str		= 'nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader'

	def init(ii):
		return 

	def launch_cmd(ii, node_i, submit_i, cmd):
		return cmd

	def end(ii):
		pass

	def wait_for_everyone(ii):
		pass

	@torch.no_grad()
	def sync(ii, v_d: dict, sync_method: str= '', this_is_noop: bool= True) -> list[Any]:
		return v_d

	def backward(ii, loss: torch.Tensor, create_graph: bool= False):
		loss.backward(create_graph=create_graph)
		
	def prepare(ii, *arg, **kw):
		from .for_torch.torch_utils import try_convert
		print('\ndist:prepare: ')
		
		kw = {k:try_convert(k, v, ii.p.device, ii.p.dtype) for k,v in kw.items()}
		return list([try_convert(v, v, ii.p.device, ii.p.dtype) for v in arg]) + list(kw.values())
	
	def set_seed(ii, seed= None) -> int:
		seed = seed or int(ii.p.seed)
		print('\ndist:set_seed: ', seed)
		seed = seed + int(ii.rank)
		torch.random.manual_seed(seed)
		return seed
		
	def set_device(ii, device= None):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print('dist:set_device: ', device)
		if not device == torch.device('cpu'):
			device_int = torch.cuda.current_device()
			torch_n_device = torch.cuda.device_count()
			cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
			print_table(dict(cuda_visible_devices=cuda_visible_devices, device_int=device_int, torch_n_device=torch_n_device))
		print('Device is ', device)
		return device
		
	def set_dtype(ii, dtype= None):
		if not dtype or ii.p.dtype:
			ii.p.dtype = 'float32'
		torch.set_default_dtype(ii.p.dtype)
		return ii.p._dtype_str

	def unwrap(ii, model):
		return model

class SingleProcess(DistBase):
	
	dist_name: str = 'SingleProcess'
	sync_every: int = property(lambda _: 0) # no sync
	head: int = property(lambda _: True)
	launch_cmd: Callable 	= property(lambda _: 
		lambda node_i, submit_i, cmd: 
		f'\nexport RANK={submit_i} \
		\necho $SLURM_JOB_NODELIST \
		\nsrun --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --ntasks=1 --exclusive --label python -u {_.p.run_name} {cmd} '
	)

class Naive(DistBase):
	
	n_launch:       int = property(lambda _: _.p.resource.n_gpu)
	n_worker:       int = property(lambda _: _.p.resource.n_gpu)
	dist_name: 	str		= 'Naive'
	sync_every:  int 	= 5
	head: 		bool 	= property(lambda _: _.rank==0 or (_.p.opt_hypam_tag in _.p.mode))
	nsync:          int = 0

	launch_cmd: Callable 	= property(lambda _: 
		lambda node_i, submit_i, cmd: 
		f'\nexport RANK={submit_i} \
		\necho $SLURM_JOB_NODELIST \
		\nsrun --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --ntasks=1 --exclusive --label python -u {_.p.run_name} {cmd} '
	)

	# bash line to access slurm job nodelist array
	# node1=$SLURM_JOB_NODELIST[0]
	"""
	- --gres=gpu:1 	: 1 gpu per task (launched with srun)
	- --cpus-per-task=8 : 8 cpus per task
	- --ntasks=8 : 8 tasks per node
	- --exclusive : exclusive use of the node
	- --label : label the output
	- --gpus-per-task=1 : 1 gpu per task"""
	
	@torch.no_grad()
	def sync(ii, v_d: dict, sync_method: str, this_is_noop: bool= False) -> list[torch.Tensor]:
		
		if this_is_noop or ii.n_worker == 1:
			return v_d
		
		if ii.nsync < 2 and os.environ.get('debug', False) in ['True', 'true']:
			ii.p.if_debug_print_d({'nsync': ii.nsync, 'this_is_noop': this_is_noop, 'ii.n_worker': ii.n_worker}, msg= 'sync Naive')
		
		v_path = (ii.p.paths.exchange_dir / f'step{ii.nsync}_rank{ii.rank}_{ii.dist_id}').with_suffix('.pk')
		v_sync_path = add_to_Path(v_path, '-sync')
		
		try:
			gc.disable()

			v_ref_leaves, treespec = optree.tree_flatten(v_d)
			v_sync_save = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in v_ref_leaves]
			dump(v_path, v_sync_save)

		except Exception as e:
			print(e)
		finally:
			gc.enable()
		
		if not ii.rank:

			n_ready = 0
			while n_ready < ii.p.resource.n_gpu:
				k_path_all = list(ii.p.paths.exchange_dir.glob(f'step{ii.nsync}_*'))
				n_ready = len(k_path_all)
				sleep(2)

			for i, p in enumerate(k_path_all):
				leaves = [load(p),] if i==0   else [*leaves, load(p)]

			v_sync = [np.stack(l) for l in zip(*leaves)]
			
			if sync_method == ii.p.mean_tag:
				v_sync = [np.mean(v, axis=0) for v in v_sync]
			elif sync_method == ii.p.gather_tag:
				pass

			try:
				gc.disable()
				for p in k_path_all:
					dump(add_to_Path(p, f'-sync'), v_sync)
			except Exception as e:
				print(e)
			finally:
				sleep(0.01)
				[p.unlink() for p in k_path_all]
				gc.enable()

		while v_path.exists():
			sleep(0.05)
		sleep(0.05)

		gc.disable()
		try:
			v_sync_leaves = load(v_sync_path)  # Speed: Only load sync vars
			v_sync_leaves = [torch.tensor(data=v, device=ref.device, dtype=ref.dtype, requires_grad=False) 
				if isinstance(ref, torch.Tensor) else v 
				for v, ref in zip(v_sync_leaves, v_ref_leaves)]
			v_sync = optree.tree_unflatten(treespec=treespec, leaves=v_sync_leaves)
			
		except Exception as e:
			v_sync = v_d
			print(e)
		finally: # ALWAYS EXECUTED
			v_sync_path.unlink()
			gc.enable()

		ii.nsync += 1
		return v_sync

HFAccelerate = None

from .core import TryImportThis
with TryImportThis('accelerate') as _hf_accel:

	import accelerate

	class HFAccelerate(DistBase):
		sync_every:  int 	= 5
		dist_name: str = 'HFAccelerate'

		class dist_c(PlugIn):
			multi_gpu = True
			machine_rank = property(lambda _: '0')
			# same_network = True
			main_process_port = property(lambda _: find_free_port()) # 
			num_processes =  property(lambda _: str(_.p.p.resource.n_gpu))
			num_machines =  property(lambda _: str(_.p.p.resource.n_node))
			num_cpu_threads_per_process = property(lambda _: str(_.p.p.resource.n_thread_per_process))

		launch_cmd:	Callable  	= property(lambda _: 
			lambda node_i, submit_i, cmd: 
			f'accelerate launch {dict_to_cmd(_.dist_c.d, exclude_false=True)} {_.p.run_name} \ "{cmd}" '
		) # ! backslash must come between run.py and cmd and "cmd" needed

		# internal
		ready: bool = property(lambda _: _.accel is not None)
		accel: accelerate.Accelerator = None
		split_batches: bool		= False  # if True, reshapes data to (n_gpu, batch_size//n_gpu, ...)
		mixed_precision: str	= 'no' # 'fp16, 'bf16'

		class hostfile(PlugIn):
			slots = 10 # !!! max per resource fix
			ip = property(lambda _: dict(
				ip0 = _.slots,
				ip1 = _.slots,
			))

		def __init__(ii, parent=None):
			super().__init__(parent=parent)
			ii.init()

		def init(ii):
			ii.accel = accelerate.Accelerator(
				split_batches=getattr(ii, 'split_batches'), mixed_precision=getattr(ii, 'mixed_precision')
			)
		
		def wait_for_everyone(ii):
			ii.accel.wait_for_everyone()

		def end(ii):
			ii.accel.wait_for_everyone()
			ii.accel.free_memory()
			try:
				ii.accel.end_training()
			except Exception as e:
				print(e)
			del ii.accel
			ii.accel = None
			print('hf_accel end. Make sure to reinit')

		@torch.no_grad()
		def sync(ii, v_d: dict[str:torch.Tensor], sync_method: str, this_is_noop: bool= False) -> list[torch.Tensor]:
			if this_is_noop or ii.n_worker==1:
				return v_d

			if sync_method == ii.p.mean_tag:
				v_sync: list[torch.Tensor] = ii.accel.reduce(v_d, reduction='mean')

			elif sync_method == ii.p.gather_tag:
				v_sync: list[torch.Tensor] = ii.accel.gather(v_d)

			else:
				raise ValueError(f'Unknown sync method {sync_method}')
			return v_sync

		def backward(ii, loss: torch.Tensor, create_graph=False):
			ii.accel.backward(loss, create_graph=create_graph)

		def set_device(ii, device= None):
			print('dist:accel: getting devices with accelerate ', ii.accel._get_devices())
			return ii.accel.device

		def set_seed(ii, seed):
			from accelerate.utils import set_seed
			print('setting seed w accelerate ', seed)
			set_seed(seed, device_specific=True)
			return seed
	
		def prepare(ii, *arg, **kw):
			print('dist:hf_accel:prepare: device,is_main,is_local', \
				ii.accel.device, ii.accel.is_main_process, ii.accel.is_local_main_process, sep='\n')
			return ii.accel.prepare(*arg, **kw) 

		def unwrap(ii, model:nn.Module):
			return ii.accel.unwrap_model(model)

		@property
		def _docs(ii,):
			"""
			# class _dist_c_multinode(PlugIn):
			# 	compute_environment: LOCAL_MACHINE
			# 	distributed_type: DEEPSPEED
			# 	fsdp_config: {}
			# 	machine_rank: 0 # 1 in the second node
			# 	main_process_ip: 192.xxx.x.xx
			# 	main_process_port: 29500
			# 	main_training_function: main
			# 	mixed_precision: fp16
			# 	num_machines: 2
			# 	num_processes: 4
			# 	use_cpu: false
				# class deepspeed_c(PlugIn):
				# 	deepspeed_multinode_launcher: standard
				# 	gradient_accumulation_steps: 1
				# 	gradient_clipping: 1.0
				# 	offload_optimizer_device: none
				# 	offload_param_device: none
				# 	zero3_init_flag: false
				# 	zero_stage: 2
				# split_batches= True
				# mixed_precision= 'no' # 'fp16, 'bf16'
			accel attrs
			**device** (torch.device) -- The device to use.
			**distributed_type** ([~utils.DistributedType]) -- The distributed training configuration.
			**local_process_index** (int) -- The process index on the current machine.
			**mixed_precision** (str) -- The configured mixed precision mode.
			**num_processes** (int) -- The total number of processes used for training.
			**optimizer_step_was_skipped** (bool) -- Whether or not the optimizer update was skipped (because of gradient overflow in mixed precision), in which case the learning rate should not be changed.
			**process_index** (int) -- The overall index of the current process among all processes.
			**state** ([~state.AcceleratorState]) -- The distributed setup state.
			**sync_gradients** (bool) -- Whether the gradients are currently being synced across all processes.
			**use_distributed** (bool) -- Whether the current configuration is for distributed training.
			"""
