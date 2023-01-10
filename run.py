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

""" 
- tmp directory 
- head node does the things 💚
	- get head node
	- get worker nodes
- functions
	- identify head node
	- get path for saving
	- list local values to share
	- 'most recent model'
	- try, except for open file
	- some notion of sync
		- dump data files, get new model, iterate again
	- data files:
		- numpy, cpu, 
		- dir: v_exchange
		- name: v_node_gpu 

- issues
	- does not work for sweep
"""

# def init_process(rank, size, fn, backend='gloo'):
# 	""" Initialize the distributed environment. """
# 	os.environ['MASTER_ADDR'] = '127.0.0.1'
# 	os.environ['MASTER_PORT'] = '29500'
# 	dist.init_process_group(backend, rank=rank, world_size=size)
# 	fn(rank, size)

# """ Gradient averaging. """
# def average_gradients(model):
# 	size = float(dist.get_world_size())
# 	for param in model.parameters():
# 		dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
# 		param.grad.data /= size

def torchify_tree(v, v_ref):
	leaves, tree_spec = optree.tree_flatten(v)
	leaves_ref, _ = optree.tree_flatten(v_ref)
	leaves = [torch.tensor(data=v, device=ref.device, dtype=ref.dtype) 
           	  if isinstance(ref, torch.Tensor) else v 
              for v, ref in zip(leaves, leaves_ref)]
	return optree.tree_unflatten(treespec=tree_spec, leaves=leaves)
		
def numpify_tree(v):
	leaves, treespec = optree.tree_flatten(v)
	leaves = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in leaves]
	return optree.tree_unflatten(treespec=treespec, leaves=leaves)

""" Distributed Synchronous SGD Example """
def run(c: Pyfig):
	torch.manual_seed(c.seed)
	torch.set_default_tensor_type(torch.DoubleTensor)   # ❗ Ensure works when default not set AND can go float32 or 64
	
	n_device = c.n_device
	print(f'🤖 {n_device} GPUs available')

	### model (aka Trainmodel) ### 
	from hwat import Ansatz_fb
	from torch import nn

	_dummy = torch.randn((1,))
	dtype = _dummy.dtype
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	c._convert(device=device, dtype=dtype)
	model: nn.Module = c.partial(Ansatz_fb).to(device).to(dtype)

	### train step ###
	from hwat import compute_ke_b, compute_pe_b
	from hwat import init_r, get_center_points

	center_points = get_center_points(c.data.n_e, c.data.a)
	r = init_r(c.data.n_b, c.data.n_e, center_points, std=0.1)
	deltar = torch.tensor([0.02]).to(device).to(dtype)
 
	print(f"""exp/actual | 
		cps    : {(c.data.n_e, 3)}/{center_points.shape}
		r      : {(c.n_device, c.data.n_b, c.data.n_e, 3)}/{r.shape}
		deltar : {(c.n_device, 1)}/{deltar.shape}
	""")

	### train ###
	import wandb
	from hwat import keep_around_points, sample_b
	from utils import compute_metrix
	
	### add in optimiser
	model.train()
	opt = torch.optim.RAdam(model.parameters(), lr=0.001)

	def train_step(model: nn.Module, r: torch.Tensor):
			
			ke = compute_ke_b(model, r, ke_method=c.model.ke_method)
			
			with torch.no_grad():
				pe = compute_pe_b(r, c.data.a, c.data.a_z)
				e = pe + ke
				e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
				e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

			model.zero_grad()
			model.requires_grad_(True)
			loss = ((e_clip - e_clip.mean())*model(r)).mean()
			loss.backward()
			model.requires_grad_(False)

			v_tr = dict(ke=ke, pe=pe, e=e, loss=loss)
			return v_tr

	for step in range(1, c.n_step+1):
		
		model.zero_grad()
		model.requires_grad_(False)
		
		with torch.no_grad():
			r, acc, deltar = sample_b(model, r, deltar, n_corr=c.data.n_corr)  # ❗needs testing 
			r = keep_around_points(r, center_points, l=5.) if step < 50 else r

		v_tr = train_step(model, r)

		if not (step % c.log_metric_step):

			with torch.no_grad():
				grads = [p.grad for p in model.parameters()]
				params = [p for p in model.parameters()]

				v_tr |= dict(acc=acc, r=r, deltar=deltar, grads=grads, params=params)
				
				v_sync = numpify_tree(v_tr)
				v_sync = c.accumulate(step, v_sync, sync=None)
				v_sync = torchify_tree(v_sync, v_tr)

				deltar = v_sync['deltar']

				model.zero_grad()
				for p, g in zip(model.parameters(), v_sync['grads']):
					p.grad += g

				if c.dist.head:
					metrix = compute_metrix(v_sync)
					wandb.log(metrix, step=step)

		opt.step()
		
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
	
	print('aqui')
	# from pyfig import slurm
	# setattr(Pyfig, 'cluster', slurm)
	c = Pyfig(wb_mode='online', arg=arg, run_sweep=False)
	
	run(c)