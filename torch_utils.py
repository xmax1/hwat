import sys
from typing import Callable
from pathlib import Path
import wandb
from pyfig_utils import PyfigBase, Sub
from functools import partial 

from utils import dict_to_cmd
from utils import flat_any

import numpy as np

import torch
import accelerate
import optree
from utils import debug_dict


def get_things(c: Pyfig, c_init: dict, **kw):

	c_init = (c_init or {}) | (kw or {})
	c.update(c_init)
	c.start()

	c.set_dtype()
	c.distribute.set_seed()
	c.distribute.set_device()
	c.to(device=c.distribute.device, dtype=c.dtype)
	debug_dict(d=c.d, msg='pyfig at execute')

	model: nn.Module = c.partial(Model)
	model.to(device=c.distribute.device, dtype=c.dtype)
	
	model_fn, params, buffers = make_functional_with_buffers(model)  
	model_fn_vmap = lambda params, _v: vmap(model_fn, in_dims=(None, None, 0))(params, buffers, _v).sum()

	opt_for_params = get_opt(**c.opt.d_flat)
	opt: torch.optim.Optimizer = opt_for_params(model.parameters())

	scheduler_for_opt: torch.optim.lr_scheduler._LRScheduler = c.partial(get_scheduler)
	scheduler: torch.optim.lr_scheduler._LRScheduler = scheduler_for_opt(opt)

	model, opt, scheduler = c.distribute.prepare(model, opt, scheduler)  # docs:accelerate

	if c.mode=='train':
		model.train()
	elif c.mode=='evaluate':
		model.eval()

	return dict(model=model, model_fn_vmap=model_fn_vmap, opt=opt, scheduler=scheduler)


class distribute(PyfigBase.distribute):
	dist: accelerate.Accelerator = None
	dist_method: 	str		= 'accelerate'
	sync_step:		int		= 5
	_launch_cmd:	str  	= property(lambda _: f'accelerate launch {dict_to_cmd(_.dist_c.d, exclude_false=True)} ')

	class dist_c(Sub):
		# compute_environment = 'LOCAL_MACHINE'
		# distributed_type =  'MULTI_GPU'
		multi_gpu = False
		machine_rank =  '0'
		same_network = True
		main_process_port = str(np.random.randint(30000, 60000))
		num_processes =  property(lambda _: str(_._p._p.resource.n_gpu))
		num_machines =  property(lambda _: str(_._p._p.resource.n_node))

	def __init__(ii, parent= None):
		super().__init__(parent)
		ii.dist: accelerate.Accelerator = accelerate.Accelerator()

	def sync(ii, v_d: dict[str:torch.Tensor], step: int) -> list[torch.Tensor]:

		if ((step/ii.sync_step)==1) and ii._p.debug:
			[print(k, v.shape) for k,v in v_d.items()]

		with torch.no_grad():
			v_flat, treespec = optree.tree_flatten(v_d)
			v_sync_flat: list[torch.Tensor] = ii.dist.gather(v_flat)
			for i, (v, v_ref) in enumerate(zip(v_sync_flat, v_flat)):
				if ((step/ii.sync_step)==1) and ii._p.debug:
					print(v.shape, v_ref.shape)
				v = v.reshape(-1, *v_ref.shape).mean(dim=0)
				v_sync_mean = [v] if i==0 else [*v_sync_mean, v]
			v_sync = optree.tree_unflatten(treespec=treespec, leaves=v_sync_mean)

		return v_sync

	def backward(ii, loss: torch.Tensor):
		ii.dist.backward(loss)

	def set_device(ii, device=None):
		print('getting devices with accelerate ', ii.dist._get_devices())
		ii.device = ii.dist.device
		return ii.device

	def set_seed(ii, seed=None):
		print('setting seed w accelerate ' )
		from accelerate.utils import set_seed
		set_seed(seed or ii._p.seed)
  
	def prepare(ii, model, opt, **kw):
		return ii.dist.prepare(model, opt, **kw)  # docs:accelerate

def gen_profile(
	fn: Callable,
	profile_dir= './dump/tmp',
	wait=1, 
	warmup=1, 
	active=1, 
	repeat=1,
	**init_d
) -> dict:
	print('profile: ', fn)

	debug_dict(d=init_d, msg='profile')

	fn = partial(fn, init_d=init_d)

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


def get_max_mem_c(fn: Callable, max_mem_min=6, max_mem_max=15, **kw) -> dict:
	t = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
	print('total memory on device: ', t)
	r = torch.cuda.memory_reserved(0)
	a = torch.cuda.memory_allocated(0)
	for n_b_power in range(max_mem_min, max_mem_max):
		try:
			n_b = 2**n_b_power
			print(f'max mem trial: n_b={n_b}, n_b_power={n_b_power}')
			v_d = fn(n_b=n_b, **kw)
			mem_used = v_d['max_mem_alloc']
			print(f'n_b {n_b} used {mem_used} out of {t}')
			torch.cuda.empty_cache()
			if mem_used > t/2:
				print('get_mem_max: b_s is ', n_b)
				return dict(n_b=n_b, n_b_max=n_b, max_mem_alloc=mem_used)
		except Exception as e:
			print('get_mem_max error: ', fn, max_mem_min, e, n_b)
			n_b = 2**(n_b_power-1)
			debug_dict(msg='mem_max-kw', d=kw)
			break

	print('get_mem_max: b_s is ', n_b)
	return dict(n_b=n_b, n_b_max=n_b)

def update_model(
	model: torch.nn.Module, 
	grads: dict[str:torch.Tensor]=None, 
	params: dict[str:torch.Tensor]=None,
	step: int = 0
):
	debug_dict(msg='model params: ', model_named=dict(model.named_parameters()))
	debug_dict(msg='update:new grads: ', grads=grads)
	debug_dict(msg='update:new params: ', params=params)

	try:
		with torch.no_grad():
			for k, p in model.named_parameters():
				if params is not None:
					p.data: torch.Tensor = params[k]

				if grads is not None:
					p.grad.data: torch.Tensor = grads[k]

	except:
		print('model, params, grads', len(grads or {}), len(params or {}), len(list(model.named_parameters())))
		grads = grads or dict(model.named_parameters())
		params = params or dict(model.named_parameters())
		for (k,v), (k_p,v_p), (k_g,v_g) in zip(model.named_parameters(), params.items(), grads.items()):
			print(k, v.shape, k_p, v_p.shape, k_g, v_g.shape)
		sys.exit('shapes')
  
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

	if opt_name == 'RAdam':
		opt_for_model = partial(torch.optim.RAdam, lr=lr)

	elif opt_name == 'Adahessian':
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
	scheduler_name: str = None,
	max_lr: float = None,
	epochs: int = None, 
	n_step: int = None,
	default: str = 'OneCycleLR',
	**kw,
) -> torch.optim.lr_scheduler._LRScheduler:

	epochs = type_check_v('epochs', epochs, int, 1)
	max_lr = type_check_v('max_lr', max_lr, float, 0.001)
	n_step = type_check_v('n_step', n_step, int, 10000)
	scheduler_name = type_check_v('scheduler_name', scheduler_name, str, '')

	epochs = epochs if epochs else 1

	if scheduler_name.lower()=='OneCycleLR'.lower():
		scheduler = partial(
			torch.optim.lr_scheduler.OneCycleLR, max_lr=max_lr, steps_per_epoch=n_step, epochs=epochs
		)

	else:
		print(f'!!! Scheduler {scheduler_name} not available, returning OneCycleLR ')
		return get_scheduler(scheduler_name=default, max_lr=max_lr, epochs=epochs, n_step=n_step)

	return scheduler
	
 
def load(c: Pyfig, path: Path=None, **things_to_load):
	load_keys = list(things_to_load.keys())
 
	path = path or c.lo_ve_path
	
	if path.suffix == '':
		step = c.step or get_max_n_from_filename(path)
		path = c.lo_ve_path / '{mode}_{group_i}_i{step}.state'.format(c.mode, c.group_i, c.step)

	state: dict = lo_ve(path=path)
	state_keys = state.keys()
	if not all([k in list(state_keys) for k in load_keys]):
		sys.exit(f'!!! tried to load {load_keys} from {state_keys} path {c.state_dir} step {c.step}')

	new_state = {}
	for k,v in things_to_load.items():
		if k in ['model', 'opt'] :
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

