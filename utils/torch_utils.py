import sys
from typing import Callable
from pathlib import Path
import wandb
from pyfig_utils import PyfigBase, PlugIn 
from functools import partial 

from utils import dict_to_cmd
from utils import flat_any

import numpy as np

import torch
import accelerate
import optree
from utils import debug_dict
from torch import nn


def gen_profile(
	fn: Callable,
	c: PyfigBase, 
	wait=1, 
	warmup=1, 
	active=1, 
	repeat=1,
	**kw
) -> dict:
	print('profile: ', fn)

	profiler = torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
		schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
		on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
		profile_memory=True, with_stack=True, with_modules=True
	)
	with profiler:
		for _ in range((wait + warmup + active) * repeat):
			fn()
			profiler.step()

	profiler.export_stacks(profile_dir/'profiler_stacks.txt', 'self_cuda_time_total')

	print(profiler.key_averages().table())

	profile_art = wandb.Artifact(f"trace", type="profile")
	p = next(profile_dir.iterdir())
	profile_art.add_file(p, "trace.pt.trace.json")
	profile_art.save()
	return init_d


def get_max_mem_c(fn: Callable, c: PyfigBase, max_mem_min=8, max_max_mem=20, **kw) -> dict:
	import traceback
	t = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
	r = torch.cuda.memory_reserved(0)
	a = torch.cuda.memory_allocated(0)
	print('get_max_mem_c:total memory on device: ', t)

	for n_b_power in range(max_mem_min, max_max_mem):
		try:
			
			n_b = n_b=2**n_b_power
			c.update(dict(n_b = n_b))

			v_run = fn(c=c)

			mem_used = v_run['max_mem_alloc']
			print(f'n_b {n_b} used {mem_used} out of {t}')

			torch.cuda.empty_cache()
			if mem_used > t/2:
				print('get_max_mem: b_s is ', n_b)
				return dict(n_b_max=n_b, max_mem_alloc=mem_used)

		except Exception as e:
			print(traceback.format_exc())
			print('v_run max_mem: ', v_run)
			raise Exception


def detach_tree(v_d: dict):
	items = []
	for k, v in v_d.items():

		if isinstance(v, list):
			v = {k + '_' + str(i): sub_i for i, sub_i in enumerate(v)}
		
		if isinstance(v, dict):
			v = {k: detach_tree(v)}
		elif isinstance(v, torch.Tensor):
			v = v.detach()
		
		items += [(k, v),]
	
	return dict(items)

  
def get_opt(
	*,
	opt_name: str = None,
	lr: float = None,
	betas: tuple[float] = None,
	eps: float = None,
	weight_decay: float = None,
	hessian_power: float = None,
	default: str = 'RAdam',
	**kw, 
) -> torch.optim.Optimizer:

	if opt_name.lower() == 'RAdam'.lower():
		opt_for_model = partial(torch.optim.RAdam, lr=lr)

	elif opt_name.lower() == 'Adahessian'.lower():
		import torch_optimizer  # pip install torch_optimizer
		opt_for_model = partial(
    		torch_optimizer.Adahessian,
			lr 			 = lr,
			betas		 = betas,
			eps			 = eps,
			weight_decay = weight_decay,
			hessian_power= hessian_power
    	)
	else:
		print(f'!!! opt {opt_name} not available, returning {default}')
		opt_for_model = get_opt(opt_name=default, lr=0.001)

	return opt_for_model


from utils import type_check_v


def get_scheduler(
	sch_name: str = None,
	sch_max_lr: float = None,
	sch_epochs: int = None, 
	n_step: int = None,
	default: str = 'OneCycleLR',
	**kw,
) -> torch.optim.lr_scheduler._LRScheduler:

	# epochs = type_check_v('epochs', epochs, int, 1)
	# max_lr = type_check_v('max_lr', max_lr, float, 0.001)
	# n_step = type_check_v('n_step', n_step, int, 10000)
	# scheduler_name = type_check_v('scheduler_name', scheduler_name, str, '')
	# epochs = epochs if epochs else 1

	if sch_name.lower() == 'OneCycleLR'.lower():
		scheduler = partial(
			torch.optim.lr_scheduler.OneCycleLR, max_lr=sch_max_lr, steps_per_epoch=n_step, epochs=sch_epochs
		)

	else:
		print(f'!!! Scheduler {sch_name} not available, returning OneCycleLR ')
		return get_scheduler(scheduler_name=default, max_lr=sch_max_lr, epochs=sch_epochs, n_step=n_step)

	return scheduler

from utils import get_max_n_from_filename
from pyfig_utils import lo_ve
 
def load(c: PyfigBase, path: Path=None, **things_to_load):

	load_keys = list(things_to_load.keys())
	path = path or c.lo_ve_path
	
	if path.suffix == '':
		step = c.step or get_max_n_from_filename(path)
		path = c.lo_ve_path / '{mode}_{group_i}_i{step}.state'.format(c.mode, c.group_i, step)

	state: dict = lo_ve(path=path)
	state_keys = state.keys()
	if not all([k in list(state_keys) for k in load_keys]):
		sys.exit(f'!!! tried to load {load_keys} from {state_keys} path {c.state_dir} step {c.step}')

	new_state = {}
	for k,v in things_to_load.items():
		if k in ['model', 'opt',] :
			v: nn.Module
			new_state[k] = v.load_state_dict(state[k])
		else:
			new_state[k] = state[k]

	return (c, *new_state.values())

# import threading, torch, time, pynvml

# def preload_pytorch():
#     torch.ones((1, 1)).cuda()

# def gpu_mem_used(id):
#     handle = pynvml.nvmlDeviceGetHandleByIndex(id)
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     return int(info.used/2**20)

# def gpu_mem_used_no_cache(id):
#     torch.cuda.empty_cache()
#     return gpu_mem_used(id)

# def peak_monitor_start():
#     global peak_monitoring
#     peak_monitoring = True

#     # this thread samples RAM usage as long as the current epoch of the fit loop is running
#     peak_monitor_thread = threading.Thread(target=peak_monitor_func)
#     peak_monitor_thread.daemon = True
#     peak_monitor_thread.start()

# def peak_monitor_stop():
#     global peak_monitoring
#     peak_monitoring = False

# def peak_monitor_func():
#     global nvml_peak, peak_monitoring
#     nvml_peak = 0
#     id = torch.cuda.current_device()

#     while True:
#         nvml_peak = max(gpu_mem_used(id), nvml_peak)
#         if not peak_monitoring: break
#         time.sleep(0.001) # 1msec


# def consume_gpu_ram(n): return torch.ones((n, n)).cuda()
# def consume_gpu_ram_256mb(): return consume_gpu_ram(2**13)

# peak_monitoring = False
# nvml_peak = 0
# preload_pytorch()
# pynvml.nvmlInit()
# id = torch.cuda.current_device()

# # push the pytorch's peak gauge high up and then release the memory
# z = [consume_gpu_ram_256mb() for i in range(4)] # 1GB
# del z

# peak_monitor_start()
# nvml_before = gpu_mem_used_no_cache(id)
# cuda_before = int(torch.cuda.memory_allocated()/2**20)

# # should be: 256 used, 512 peaked
# c1 = consume_gpu_ram_256mb()
# c2 = consume_gpu_ram_256mb()
# del c1

# # code finished
# peak_monitor_stop()
# nvml_after = gpu_mem_used_no_cache(id)
# cuda_after = int(torch.cuda.memory_allocated()/2**20)
# cuda_peak  = int(torch.cuda.max_memory_allocated()/2**20)
# print("nvml:", nvml_after-nvml_before, nvml_peak-nvml_before)
# print("cuda:", cuda_after-cuda_before, cuda_peak-cuda_before)

# if c.model.compile_ts:
# 		# os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
# 		model = torch.jit.script(model, r.clone())
		
# 	if c.model.compile_func:
# 		pass
# 		# https://pytorch.org/tutorials/intermediate/nvfuser_intro_tutorial.html
  
# 	if c.model.optimise_ts:
# 		pass
# 		# from torch.utils.mobile_optimizer import optimize_for_mobile
# 		# optimized_torchscript_model = optimize_for_mobile(torchscript_model)
# 		# The optimized model can then be saved and deployed in mobile apps:
# 		# optimized_torchscript_model.save("optimized_torchscript_model.pth")

# 	if c.model.optimise_aot:
# 		# https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html
# 		pass

# 	if c.model.functional:
# 		pass

# torch.backends.cudnn.benchmark = True
# torch.manual_seed(c.seed)

# print('__Python VERSION:', sys.version)
# print('__pyTorch VERSION:', torch.__version__)
# print('__CUDA VERSION')
# from subprocess import call
# # call(["nvcc", "--version"]) does not work
# ! nvcc --version
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())
# print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# print('Active CUDA Device: GPU', torch.cuda.current_device())
# print ('Available devices ', torch.cuda.device_count())
# print ('Current cuda device ', torch.cuda.current_device())

# from subprocess import call
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])

