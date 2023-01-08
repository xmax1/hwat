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
		

		

""" Distributed Synchronous SGD Example """
def run(c: Pyfig):
	torch.manual_seed(c.seed)
	torch.set_default_tensor_type(torch.DoubleTensor)   # ❗ Ensure works when default not set AND can go float32 or 64
	
	n_device = c.n_device
	print(f'🤖 {n_device} GPUs available')

	### model (aka Trainmodel) ### 
	from hwat import Ansatz_fb

	_dummy = torch.randn((1,))
	dtype = _dummy.dtype
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	c._convert(device=device, dtype=dtype)
	model = c.partial(Ansatz_fb).to(device).to(dtype)

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
	opt = torch.optim.RAdam(model.parameters(), lr=0.01)

	def train_step(model, r):

			ke = compute_ke_b(model, r, ke_method=c.model.ke_method).detach()
			
			with torch.no_grad():
				pe = compute_pe_b(r, c.data.a, c.data.a_z)
				e = pe + ke
				e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
				e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

			opt.zero_grad()
			loss = ((e_clip - e_clip.mean())*model(r)).mean()
			loss.backward()
			opt.step()
   
			grads = [p.grad.detach() for p in model.parameters()]
			params = [p.detach() for p in model.parameters()]

			v_tr = dict(ke=ke, pe=pe, e=e, loss=loss, params=params, grads=grads)
			return v_tr

	if c.head: 
		wandb.define_metric("*", step_metric="tr/step")
	 
	for step in range(1, c.n_step+1):
	 
		r, acc, deltar = sample_b(model, r, deltar, n_corr=c.data.n_corr)  # ❗needs testing 
		r = keep_around_points(r, center_points, l=5.) if step < 50 else r

		v_tr = train_step(model, r)
		v_tr |= dict(acc=acc, r=r, deltar=deltar)
		
		if not (step % c.log_metric_step):
			metrix = compute_metrix(v_tr.copy())  # ❗ needs converting to torch, ie tree maps
			v_tr = c.accumulate(step, v_tr)
			
			def torchify_tree(v):
				if np.isscalar(v):
					return v
				import optree
				leaves, tree_spec = optree.tree_flatten(v)
				leaves = [torch.Tensor(v) for v in leaves]
				return optree.tree_unflatten(tree_spec, leaves)

			v_tr = {k:torchify_tree(v) for k,v in v_tr.items()}
			for p torch.Tensor, p_new in zip(model.parameters(), p_new):
				p = p_new.requires_grad(True)

			if c.head:
				wandb.log({'tr/step':step, **metrix})
		

		# if not (step % c.dist.accumulate_step):
	  		
		# print_keys = ['e']
		# pprint.pprint(dict(step=step) | {k:v.mean() 
		# 	if isinstance(v, torch.Tensor) else v for k,v in v_tr.items() if k in print_keys})
		
		if not (step-1):
			print('End Of 1')
   
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
		n_corr = 50, 
		n_step = 2000, 
		log_metric_step = 10, 
		exp_name = 'demo',
		# sweep = {},
	)
	
	c = Pyfig(wb_mode='online', arg=arg, submit=False, run_sweep=False)
	
	run(c)