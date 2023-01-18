# TORCH MNIST DISTRIBUTED EXAMPLE

"""run.py:"""
#!/usr/bin/env python
import os
from accelerate import Accelerator
import torch

from accelerate.utils import set_seed
from pyfig import Pyfig
import numpy as np
import pprint
from pathlib import Path
from time import sleep
import shutil
import optree
from copy import deepcopy
# https://huggingface.co/docs/accelerate/basic_tutorials/migration


def run(c: Pyfig):
	dist = Accelerator()
	set_seed(c.seed)
	print(dist)
	# torch.backends.cudnn.benchmark = True
	torch.manual_seed(c.seed)
	torch.set_default_tensor_type(torch.DoubleTensor)   # ❗ Ensure works when default not set AND can go float32 or 64

	n_device = c.resource.n_device
	print(f'🤖 {n_device} GPUs available')

	### model (aka Trainmodel) ### 
	from hwat import Ansatz_fb
	from torch import nn

	_dummy = torch.randn((1,))
	dtype = _dummy.dtype
	# device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = dist.device
	c.to(to='torch', device=device, dtype=dtype)

	### init things ###
	from hwat import compute_ke_b, compute_pe_b
	from hwat import init_r, get_center_points
	from functorch import make_functional_with_buffers, vmap
	from copy import deepcopy
 
	center_points = get_center_points(c.data.n_e, c.data.a)
	r = init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
	deltar = torch.tensor([0.02]).to(device).to(dtype)

	model: nn.Module = c.partial(Ansatz_fb).to(device).to(dtype)
	model.requires_grad_(False)
	model_og = deepcopy(model)

	model_fn, params, buffers = make_functional_with_buffers(model_og)  
	model_rv = lambda params, _r: vmap(model_fn, in_dims=(None, None, 0))(params, buffers, _r).sum()

	if c.model.compile_ts:
		# os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
		model = torch.jit.script(model, r.clone())
		
	if c.model.compile_func:
		pass
		# https://pytorch.org/tutorials/intermediate/nvfuser_intro_tutorial.html
  
	if c.model.optimise_ts:
		pass
		# from torch.utils.mobile_optimizer import optimize_for_mobile
		# optimized_torchscript_model = optimize_for_mobile(torchscript_model)
		# The optimized model can then be saved and deployed in mobile apps:
		# optimized_torchscript_model.save("optimized_torchscript_model.pth")

	if c.model.optimise_aot:
		# https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html
		pass

	if c.model.functional:
		pass
 
	### train step ###
	
	print(f"""exp/actual | 
		cps    : {(c.data.n_e, 3)}/{center_points.shape}
		r      : {(c.resource.n_device, c.data.n_b, c.data.n_e, 3)}/{r.shape}
		deltar : {(c.resource.n_device, 1)}/{deltar.shape}
	""")

	### train ###
	import wandb
	from hwat import keep_around_points, sample_b
	from utils import compute_metrix

	def get_opt(c: Pyfig):
		if c.opt.opt_name == 'RAdam':
			return torch.optim.RAdam(model.parameters(), lr=c.opt.lr)

		if c.opt.opt_name == 'Adahessian':
			import torch_optimizer  # pip install torch_optimizer
			return torch_optimizer.Adahessian(
						model.parameters(),
						lr 			= c.opt.lr,
						betas		= c.opt.betas,
						eps			= c.opt.eps,
						weight_decay= c.opt.weight_decay,
	  					hessian_power= c.opt.hessian_power)
		raise NotImplementedError

	opt = get_opt(c)

	model, opt = dist.prepare(model, opt)
	# model, opt = c.setup_distribute(model, opt)

	opt: torch.optim.Optimizer
	model: torch.nn.Module

	for step in range(1, c.n_step+1):
		
		opt.zero_grad()

		for p in model.parameters():
			p.requires_grad = False

		r, acc, deltar = sample_b(model, r, deltar, n_corr=c.data.n_corr) 
		r = keep_around_points(r, center_points, l=5.) if step < 50 else r
  
		ke = compute_ke_b(model, model_rv, r, ke_method=c.model.ke_method)
		pe = compute_pe_b(r, c.data.a, c.data.a_z)
		
		e = pe + ke
		e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
		e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

		opt.zero_grad()
		for p in model.parameters():
			p.requires_grad = True
		loss: torch.Tensor = ((e_clip - e_clip.mean())*model(r)).mean()
		
		dist.backward(loss)

		params = [p.data for p in model.parameters()]
		grads = [p.grad.data for p in model.parameters()]
		
		v_tr = dict(acc=acc, r=r, deltar=deltar, e=e, pe=pe, ke=ke, grads=grads, params=params)

		if not (step % c.log_metric_step):

			if c.resource.n_gpu > 1:
				def sync(v_tr):
					return dist.gather(v_tr)
				v_tr = sync(v_tr)
				# v_tr = c.sync(step, v_tr)
				deltar = v_tr['deltar']

			if c.distribute.rank:
				metrix = compute_metrix(v_tr)
				wandb.log(metrix, step=step)
		
		# with torch.no_grad():
		# 	for p, g in zip(model.parameters(), v_tr['grads']):
		# 		p.grad = g
		opt.step()

		if not (step-1) and c.distribute.rank and c.profile and not c.distribute.dist_method == 'accelerate':
			
			def gen_profile(wait=1, warmup=1, active=1, repeat=1):
				# https://wandb.ai/wandb/trace/reports/Using-the-PyTorch-Profiler-with-W-B--Vmlldzo5MDE3NjU
				print('profiling')
				model_tmp = deepcopy(model)
				model_pr = lambda : model_tmp(r)
				sample_pr = lambda : sample_b(model_tmp, r, deltar, n_corr=c.data.n_corr) 
				ke_pr = lambda : compute_ke_b(model_tmp, model_rv, r, ke_method=c.model.ke_method)

				profiler = torch.profiler.profile(
					schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
					on_trace_ready=torch.profiler.tensorboard_trace_handler(c.profile_dir),
					profile_memory=True, with_stack=True, with_modules=True
				)
				with profiler:
					times = dict(t_model=0.0, t_sample=0.0, t_ke=0.0)
					import time
					for _ in range((wait + warmup + active) * repeat):
						t0 = time.time()
						model_pr()
						times['t_model'] += time.time() - t0
						t0 = time.time()
						sample_pr()
						times['t_sample'] += time.time() - t0
						t0 = time.time()
						ke_pr()
						times['t_ke'] += time.time() - t0
						profiler.step()
				for k,v in times.items():
					c.wb.run.summary[k] = v

				profile_art = wandb.Artifact(f"trace", type="profile")
				for p in c.profile_dir.iterdir():
					profile_art.add_file(p, "trace.pt.trace.json")
					break
				profile_art.save()
				profiler.export_stacks(c.profile_dir/'profiler_stacks.txt', 'self_cuda_time_total')
				"""
				# if not c.model.compile_ts: # https://github.com/pytorch/pytorch/issues/76791
				# docs:profiler
				1- --> wandb --> Artifacts --> files --> trace
				https://wandb.ai/wandb/trace/reports/Using-the-PyTorch-Profiler-with-W-B--Vmlldzo5MDE3NjU
				2- tensorboard --logdir=c.profile_dir
				browser: http://localhost:6006/pytorch_profiler
				https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
				"""
				print('profile end.')

			gen_profile()

if __name__ == "__main__":
	
	arg = dict(
		charge = 0,
		spin  = 0,
		a = np.array([[0.0, 0.0, 0.0],]),
		a_z  = np.array([6.,]),
		n_b = 1024, 
		n_sv = 32, 
		n_pv = 16,
		n_det = 1,
		n_corr = 10, 
		n_step = 5000, 
		log_metric_step = 20, 
		exp_name = 'demo',
	)
	sweep_base = dict(method= 'grid')
 
	# # exp 1
	# sweep_arg = dict()
	# sweep = dict(parameters= dict(
	# 	n_b  = dict(values=[512, 1024, 2048, 4096, 8192]),
	# ))

	# exp 2	
	# sweep_arg = dict()
	# sweep = dict(parameters= dict(
	# 	n_det  = dict(values=[1, 4, 8,]),
	# ))

	# sweep_arg = dict(system='O2_neutral_triplet')
	# sweep = dict(parameters= dict(
	# 	n_b  = dict(values=[256, 512, 1024, 2048]),
	# ))

	sweep_arg = dict()
	sweep = dict(parameters= dict(
		a_z  = dict(values=[np.array([float(i),]) for i in range(40)]),
	))
	c = Pyfig(init_arg=arg | sweep_arg, sweep=sweep)
	run(c)
