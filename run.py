# TORCH MNIST DISTRIBUTED EXAMPLE

"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pyfig import Pyfig
import numpy as np

def init_process(rank, size, fn, backend='gloo'):
	""" Initialize the distributed environment. """
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'
	dist.init_process_group(backend, rank=rank, world_size=size)
	fn(rank, size)

""" Gradient averaging. """
def average_gradients(model):
	size = float(dist.get_world_size())
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
		param.grad.data /= size
		
""" Distributed Synchronous SGD Example """
def run(c: Pyfig):
	torch.manual_seed(1234)
	torch.set_default_tensor_type(torch.DoubleTensor)   # ❗ Ensure works when default not set AND can go float32 or 64
	
	n_device = c.n_device
	print(f'🤖 {n_device} GPUs available')

	### model (aka Trainmodel) ### 
	from hwat_func import Ansatz_fb
	from torch import nn

	_dummy = torch.randn((1,))
	dtype = _dummy.dtype
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	c._convert(device=device, dtype=dtype)
	model = c.partial(Ansatz_fb).to(device).to(dtype)

	
	### train step ###
	from hwat_func import compute_ke_b, compute_pe_b
	from hwat_func import init_r, get_center_points

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
	from hwat_func import keep_around_points, sample_b
	from utils import compute_metrix
	
	### add in optimiser
	# model.train()
	opt = torch.optim.RAdam(model.parameters(), lr=0.01)
	
	### fix sampler
	### fix train step 
	### metrix conversion
	from functorch import vmap, make_functional, grad
	
	model_fn, params = make_functional(model)
	# model_v = torch.compile(model_fn)
	model_v = vmap(model_fn, in_dims=(None, 0))
	

	def time_step(fn):
		# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
		from torch.profiler import profile, record_function, ProfilerActivity
		with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
			with record_function("model_inference"):
				fn()
		print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))



	def train_step(model, r):

			params = [p.detach() for p in model.parameters()]
   
			with torch.no_grad():
				
				model_ke = lambda _r: model_v(params, _r).sum()

				ke = compute_ke_b(model_ke, r)
				pe = compute_pe_b(r, c.data.a, c.data.a_z)
				e = pe + ke
				e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
				e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)

			opt.zero_grad()
			loss = ((e_clip - e_clip.mean())*model_v(model.parameters(), r)).mean()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
			loss.backward()
   
			grads = [p.grad.detach() for p in model.parameters()]
			# params = params.grad.data.add_(cfg.learning_rate * params.grad.data)
			
			# loss_fn = lambda _params: ((e_clip - e_clip.mean())*model_v(_params, r)).mean()
			# grads = grad(loss_fn)(params)
			# for p, g in zip(model.parameters(), grads):
			#     p.grad.data = g.clone()
			#     torch.nn.utils.clip_grad_norm_(p.grad, max_norm=1.)

			opt.step()
   
			v_tr = dict(
				ke=ke, pe=pe, e=e, loss=loss, params=params, grads=grads
			)
			return v_tr

	v_tr = dict(params=params, r=r, deltar=deltar)

	wandb.define_metric("*", step_metric="tr/step")
	for step in range(1, c.n_step+1):
     
		r, acc, deltar = sample_b(model_v, v_tr['params'], r, deltar, n_corr=c.data.n_corr)  # ❗needs testing 
		r = keep_around_points(r, center_points, l=5.) if step < 50 else r

		v_tr = train_step(model, r)
		
		if not (step % c.log_metric_step):
			v_tr |= dict(acc=acc, r=r, deltar=deltar)
			metrix = compute_metrix(v_tr.copy())  # ❗ needs converting to torch, ie tree maps
			wandb.log({'tr/step':step, **metrix})
		
		if not (step-1):
			print('End Of 1')
			
	# for epoch in range(10):
	#     epoch_loss = 0.0
	#     for data, target in train_set:
	#         optimizer.zero_grad()
	#         output = model(data)
	#         loss = F.nll_loss(output, target)
	#         epoch_loss += loss.item()
	#         loss.backward()
	#         average_gradients(model)
	#         optimizer.step()
	#     print('Rank ', dist.get_rank(), ', epoch ',
	#           epoch, ': ', epoch_loss / num_batches)





if __name__ == "__main__":
	
	### pyfig ###
	arg = dict(
		charge = 0,
		spin  = 0,
		a = np.array([[0.0, 0.0, 0.0],]),
		a_z  = np.array([4.,]),
		n_b = 2, 
		n_sv = 32, 
		n_pv = 16, 
		n_corr = 40, 
		n_step = 10000, 
		log_metric_step = 5, 
		exp_name = 'demo',
		# sweep = {},
	)
	
	c = Pyfig(wb_mode='online', arg=arg, submit=False, run_sweep=False)
	
	run(c)
	### DISTRIBUTED   # ❗# ❗# ❗# ❗ after single gpu demo
	# size = 2
	# processes = []
	# mp.set_start_method("spawn")
	# for rank in range(size):
	#     p = mp.Process(target=init_process, args=(rank, size, run))
	#     p.start()
	#     processes.append(p)

	# for p in processes:
	#     p.join()
		
		
		
	# fun stuff after 
	
	""" live plotting in another notebook """
	""" copy lines and run in analysis while the exp is live """
	# api = wandb.Api()
	# run = api.run("<run-here>")
	# c = run.config
	# h = run.history()
	# s = run.summary
	
	# ### fancy logging variables, philosophically reminding us of the goal ###
	# fancy = dict(
	#         pe		= r'$V(X)',    				
	#         ke		= r'$\nabla^2',    		
	#         e		= r'$E',						
	#         log_psi	= r'$\log\psi', 			
	#         deltar	= r'$\delta_\mathrm{r}',	
	#         x		= r'$r_\mathrm{e}',
	# )

