# TORCH MNIST DISTRIBUTED EXAMPLE

"""run.py:"""
#!/usr/bin/env python
import os
import torch

from pyfig import Pyfig
import numpy as np
import pprint
from pathlib import Path
from time import sleep
import shutil
import optree
from copy import deepcopy

""" Distributed Synchronous SGD Example """
def run(c: Pyfig):
	torch.manual_seed(c.seed)
	torch.set_default_tensor_type(torch.DoubleTensor)   # ❗ Ensure works when default not set AND can go float32 or 64
	
	n_device = c.resource.n_device
	print(f'🤖 {n_device} GPUs available')

	### model (aka Trainmodel) ### 
	from hwat import Ansatz_fb
	from torch import nn

	_dummy = torch.randn((1,))
	dtype = _dummy.dtype
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	c.to(to='torch', device=device, dtype=dtype)

	### init things ###
	from hwat import compute_ke_b, compute_pe_b
	from hwat import init_r, get_center_points

	center_points = get_center_points(c.data.n_e, c.data.a)
	r = init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
	deltar = torch.tensor([0.02]).to(device).to(dtype)

	model: nn.Module = c.partial(Ansatz_fb).to(device).to(dtype)

	if c.model.compile_ts:
		from utils import run_cmds
		# PYTORCH_JIT=0  # disable jit
		# run_cmds('export PYTORCH_NVFUSER_DISABLE=fallback')
		# run_cmds(['PYTORCH_NVFUSER_DISABLE_FALLBACK=1', 'export PYTORCH_NVFUSER_DISABLE_FALLBACK'], silent=False)
		# @sjlee0407 The issue you are encountering 	
		# is because you have allreduce_post_accumulation=False, allreduce_post_accumulation_fp16=False
		# Torchscript/NVFuser currently works with the above two flags set to true. 
		# Setting the above two to true will also increase performance orthogonally.
		os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
		model_og = deepcopy(model)
		model(r)
		model = torch.jit.script(model, r)
		print(type(model))
		# raise NotImplementedError

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
	def train_step(model, r: torch.Tensor):

		ke = compute_ke_b(model, r, ke_method=c.model.ke_method)
		
		with torch.no_grad():
			pe = compute_pe_b(r, c.data.a, c.data.a_z)
			e = pe + ke
			e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
			e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

		model.zero_grad()
		loss = ((e_clip - e_clip.mean())*model(r)).mean()
		loss.backward()

		return dict(ke=ke, pe=pe, e=e, loss=loss)

	print(f"""exp/actual | 
		cps    : {(c.data.n_e, 3)}/{center_points.shape}
		r      : {(c.resource.n_device, c.data.n_b, c.data.n_e, 3)}/{r.shape}
		deltar : {(c.resource.n_device, 1)}/{deltar.shape}
	""")

	### train ###
	import wandb
	from hwat import keep_around_points, sample_b
	from utils import compute_metrix
	
	opt = torch.optim.RAdam(model.parameters(), lr=0.001)

	for step in range(1, c.n_step+1):
     
		opt.step()
		model.zero_grad()
		
		with torch.no_grad():
			r, acc, deltar = sample_b(model, r, deltar, n_corr=c.data.n_corr) 
			r = keep_around_points(r, center_points, l=5.) if step < 50 else r

		v_tr = train_step(model, r)

		if not (step % c.log_metric_step):

			with torch.no_grad():
				grads = [p.grad for p in model.parameters()]
				params = [p for p in model.parameters()]

				v_tr |= dict(acc=acc, r=r, deltar=deltar, grads=grads, params=params)

				if c.resource.n_gpu > 1:
					v_tr = c.sync(step, v_tr)
					deltar = v_tr['deltar']

				model.zero_grad()
				for p, g in zip(model.parameters(), v_tr['grads']):
					p.grad += g

				if c.distribute.head:
					metrix = compute_metrix(v_tr)
					wandb.log(metrix, step=step)
  
		if not (step-1) and c.distribute.head:

			def gen_profile(wait=1, warmup=1, active=3, repeat=2):
				print('profiling')
				model_tmp = deepcopy(model)
				model_pr = lambda : model_tmp(r)
				sample_pr = lambda : sample_b(model_tmp, r, deltar, n_corr=c.data.n_corr) 
				ke_pr = lambda : compute_ke_b(model_tmp, r, ke_method=c.model.ke_method)

				with torch.profiler.profile(
					schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
					on_trace_ready=torch.profiler.tensorboard_trace_handler(c.profile_dir),
					record_shapes=True, profile_memory=True, with_stack=True
				) as prof:

					for _ in range((wait + warmup + active) * repeat):
						model_pr()
						sample_pr()
						ke_pr()
						prof.step()

				profile_art = wandb.Artifact(f"trace", type="profile")
				for p in c.profile_dir.iterdir():
					profile_art.add_file(p)
				profile_art.save()
				print('profile end.')

			gen_profile()

if __name__ == "__main__":
	
	### pyfig ###
	arg = dict(
		charge = 0,
		spin  = 0,
		a = np.array([[0.0, 0.0, 0.0],]),
		a_z  = np.array([4.,]),
		n_b = 256, 
		n_sv = 32, 
		n_pv = 16,
		n_det = 1,
		n_corr = 10, 
		n_step = 2000, 
		log_metric_step = 5, 
		exp_name = 'demo',
		# sweep = {},
	)
	
	# from pyfig import slurm
	# setattr(Pyfig, 'cluster', slurm)
	c = Pyfig()
	
	run(c)