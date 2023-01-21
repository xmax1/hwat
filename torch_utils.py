
from typing import Callable
import wandb
from pyfig_utils import PyfigBase, Sub

from utils import dict_to_cmd
from utils import flat_any

import numpy as np

import torch
import accelerate
import optree


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

	def __init__(ii, parent=None):
		super().__init__(parent)
		ii.dist: accelerate.Accelerator = accelerate.Accelerator()

	def sync(ii, v_d: torch.Tensor, step: int) -> list[torch.Tensor]:

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
		device = ii.dist.device
		ii._p.update({k:v.to(device) for k,v in flat_any(ii._p.d).items() if isinstance(v, torch.Tensor)})
		return device

	def set_seed(ii, seed=None):
		print('setting seed w accelerate ' )
		from accelerate.utils import set_seed
		set_seed(ii._p.seed)
  
	def prepare(ii, model, opt, **kw):
		return ii.dist.prepare(model, opt, **kw)  # docs:accelerate


def gen_profile(
    fn: Callable,
	c = None, 
    wait=1, warmup=1, active=1, repeat=1,
):

	profiler = torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
		schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
		on_trace_ready=torch.profiler.tensorboard_trace_handler(c.profile_dir),
		profile_memory=True, with_stack=True, with_modules=True
	)
	with profiler:
		for _ in range((wait + warmup + active) * repeat):
			fn()
			profiler.step()

	profiler.export_stacks(c.profile_dir/'profiler_stacks.txt', 'self_cuda_time_total')

	print(profiler.key_averages().table())

	profile_art = wandb.Artifact(f"trace", type="profile")
	p = next(c.profile_dir.iterdir())
	profile_art.add_file(p, "trace.pt.trace.json")
	profile_art.save()


def get_max_mem_c(fn: Callable, r: torch.Tensor, deltar: torch.Tensor, start=5, **kw):
	print('finding max_mem')
	n_b_0 = r.shape[0]
	for n_b_power in range(start, 15):
		try:
			n_b = 2**n_b_power
			n_factor = max(1, n_b // n_b_0)
			r_mem = r.tile((n_factor,) + (1,)*(r.ndim-1))
			assert r_mem.shape[0] == n_b
			stats = gen_profile(fn, r=r, deltar=deltar, mode='train', c_update=dict(n_b=n_b))
			print(stats)
		except Exception as e:
			print('mem_max: ', e)
		finally:
			n_b_max = 2**(n_b_power-1)
			return dict(c_update=dict(n_b=n_b_max))




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

