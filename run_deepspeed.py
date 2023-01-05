# TORCH MNIST DISTRIBUTED EXAMPLE

"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pyfig import Pyfig
import numpy as np
import pprint
import deepspeed
		
""" Distributed Synchronous SGD Example """
def run(c: Pyfig):
	
	torch.manual_seed(1234)
	torch.set_default_tensor_type(torch.DoubleTensor)   # ❗ Ensure works when default not set AND can go float32 or 64
	
	n_device = c.n_device
	print(f'🤖 {n_device} GPUs available')

	### model (aka Trainmodel) ### 
	from hwat_b import Ansatz_fb
	from torch import nn

	_dummy = torch.randn((1,))
	dtype = _dummy.dtype
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	c._convert(device=device, dtype=dtype)
	model = c.partial(Ansatz_fb).to(device).to(dtype)

	### train step ###
	from hwat_b import compute_ke_b, compute_pe_b
	from hwat_b import init_r, get_center_points

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
	from hwat_b import keep_around_points, sample_b
	from utils import compute_metrix
	import json 
 
	ds_config = json.load(
	"""
	{"train_batch_size": 2048,
	"gradient_accumulation_steps": 1,
	"steps_per_print": 10,
	"optimizer": {
		"type": "AdamW",
		"params": {
			"lr": 5e-05
		}
		},
		"fp16": {
			"enabled": false
		}
	}
	""")	
 
	from hwat_b import fb_block, logabssumdet
	from typing import Final

	class ModelSample(nn.Module):
	
		n_e: Final[int]                 # number of electrons
		n_u: Final[int]                 # number of up electrons
		n_d: Final[int]                 # number of down electrons
		n_det: Final[int]               # number of determinants
		n_fb: Final[int]                # number of feedforward blocks
		n_pv: Final[int]                # latent dimension for 2-electron
		n_sv: Final[int]                # latent dimension for 1-electron
		a: torch.Tensor      
		n_a: Final[int]                 # nuclei positions
		with_sign: Final[bool]          # return sign of wavefunction

		def __init__(ii, n_e, n_u, n_d, n_det, n_fb, n_pv, n_sv, a: torch.Tensor, with_sign=False):
			super(ModelSample, ii).__init__()
			ii.n_e = n_e                  # number of electrons
			ii.n_u = n_u                  # number of up electrons
			ii.n_d = n_d                  # number of down electrons
			ii.n_det = n_det              # number of determinants
			ii.n_fb = n_fb                # number of feedforward blocks
			ii.n_pv = n_pv                # latent dimension for 2-electron
			ii.n_sv = n_sv                # latent dimension for 1-electron
			ii.a = a       
			ii.n_a = len(a)               # nuclei positions
			ii.with_sign = with_sign      # return sign of wavefunction

			ii.n1 = [4*ii.n_a,] + [ii.n_sv,]*(ii.n_fb+1)
			ii.n2 = [4,] + [ii.n_pv,]*(ii.n_fb+1)

			ii.Vs = nn.ModuleList([
				nn.Linear(3*ii.n1[i]+2*ii.n2[i], ii.n1[i+1]) for i in range(ii.n_fb)
			])
			ii.Ws = nn.ModuleList([
				nn.Linear(ii.n2[i], ii.n2[i+1]) for i in range(ii.n_fb)
			])

			ii.V_u_after = nn.Linear(3*ii.n1[-1]+2*ii.n2[-1], ii.n1[-1])
			ii.V_d_after = nn.Linear(3*ii.n1[-1]+2*ii.n2[-1], ii.n1[-1])

			ii.wu = nn.Linear(ii.n_sv, ii.n_u * ii.n_det)
			ii.wd = nn.Linear(ii.n_sv, ii.n_d * ii.n_det)

			ii.r = r
			ii.deltar = deltar
			ii.acc = 0.
	
		def wf(ii, r: torch.Tensor):
			dtype, device = r.dtype, r.device

			if len(r.shape) == 2:
				r = r.reshape(-1, ii.n_e, 3) # (n_batch, n_e, 3)
			
			n_batch = r.shape[0]
	
			eye = torch.eye(ii.n_e, device=device, dtype=dtype).unsqueeze(-1)

			ra = r[:, :, None, :] - ii.a[:, :] # (n_batch, n_e, n_a, 3)
			ra_len = torch.linalg.norm(ra, dim=-1, keepdim=True) # (n_batch, n_e, n_a, 1)

			rr = r[:, None, :, :] - r[:, :, None, :] # (n_batch, n_e, n_e, 1)
			rr_len = torch.linalg.norm(rr + eye, dim=-1, keepdim=True) #* (torch.ones((ii.n_e, ii.n_e, 1), device=device, dtype=dtype)-eye) # (n_batch, n_e, n_e, 1)

			s_v = torch.cat([ra, ra_len], dim=-1).reshape(n_batch, ii.n_e, -1) # (n_batch, n_e, n_a*4)
			p_v = torch.cat([rr, rr_len], dim=-1) # (n_batch, n_e, n_e, 4)
			
			s_v_block = fb_block(s_v, p_v, ii.n_u, ii.n_d)
			
			for l, (V, W) in enumerate(zip(ii.Vs, ii.Ws)):
				s_v = torch.tanh(V(s_v_block)) + (s_v if (s_v.shape[-1]==ii.n_sv) else torch.zeros((n_batch, ii.n_e, ii.n_sv), device=device, dtype=dtype)) # (n_batch, n_e, n_sv)
				p_v = torch.tanh(W(p_v)) + (p_v if (p_v.shape[-1]==ii.n_pv) else torch.zeros((n_batch, ii.n_e, ii.n_e, ii.n_pv), device=device, dtype=dtype)) # (n_batch, n_e, n_e, n_pv)
				s_v_block = fb_block(s_v, p_v, ii.n_u, ii.n_d) # (n_batch, n_e, 3n_sv+2n_pv)

			s_u, s_d = torch.split(s_v_block, [ii.n_u, ii.n_d], dim=1) # (n_batch, n_u, 3n_sv+2n_pv), (n_batch, n_d, 3n_sv+2n_pv)

			s_u = torch.tanh(ii.V_u_after(s_u)) # (n_batch, n_u, n_sv)    # n_sv//2)
			s_d = torch.tanh(ii.V_d_after(s_d)) # (n_batch, n_d, n_sv)    # n_sv//2)

			# Map to orbitals for multiple determinants
			s_wu = torch.cat(ii.wu(s_u).unsqueeze(1).chunk(ii.n_det, dim=-1), dim=1) # (n_batch, n_det, n_u, n_u)
			s_wd = torch.cat(ii.wd(s_d).unsqueeze(1).chunk(ii.n_det, dim=-1), dim=1) # (n_batch, n_det, n_d, n_d)
			assert s_wd.shape == (n_batch, ii.n_det, ii.n_d, ii.n_d)

			ra_u, ra_d = torch.split(ra, [ii.n_u, ii.n_d], dim=1) # (n_batch, n_u, n_a, 3), (n_batch, n_d, n_a, 3)

			exp_u = torch.linalg.norm(ra_u, dim=-1, keepdim=True) # (n_batch, n_u, n_a, 1)
			exp_d = torch.linalg.norm(ra_d, dim=-1, keepdim=True) # (n_batch, n_d, n_a, 1)

			assert exp_d.shape == (n_batch, ii.n_d, ii.a.shape[0], 1)

			# print(torch.exp(-exp_u).sum(axis=2).unsqueeze(1).shape) # (n_batch, 1, n_u, 1)
			orb_u = (s_wu * (torch.exp(-exp_u).sum(axis=2).unsqueeze(1))) # (n_batch, n_det, n_u, n_u)
			orb_d = (s_wd * (torch.exp(-exp_d).sum(axis=2).unsqueeze(1))) # (n_batch, n_det, n_d, n_d)

			assert orb_u.shape == (n_batch, ii.n_det, ii.n_u, ii.n_u)

			log_psi, sgn = logabssumdet(orb_u, orb_d)

			if ii.with_sign:
				return log_psi.squeeze(), sgn.squeeze()
			else:
				return log_psi.squeeze()
			
		def compute_grads(ii, step: int):
			with torch.no_grad():
				
				ii.r, ii.acc, ii.deltar = sample_b(ii.wf, ii.r, ii.deltar, n_corr=c.data.n_corr)  # ❗needs testing 
				r = keep_around_points(ii.r, center_points, l=5.) if step < 50 else ii.r
	
				ke = compute_ke_b(ii.wf, ii.r, ke_method=c.model.ke_method)
				pe = compute_pe_b(ii.r, c.data.a, c.data.a_z)
				e = pe + ke
				e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
				e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)
	
			return e_clip - e_clip.mean()

		def forward(ii, step: int):
			energy_term = ii.compute_grads(step)
			return (energy_term * ii.wf(ii.r)).mean()

	model_sample = ModelSample(model)

	print(model_sample.parameters())
 
	model_engine, optimizer, _, _ = \
		deepspeed.initialize(args=None, model=model, model_parameters=model_sample.parameters()) # , config=c.deepspeed_c)

	model_sample(r)

	wandb.define_metric("*", step_metric="tr/step")
	for _ in range(1, c.n_step+1):
		
		loss = model_sample()
		model_engine.backward(loss)
		model_engine.step()

		print('Loss: ', loss.mean())
  
		# if not (step % c.log_metric_step):
		# 	v_tr |= dict(acc=acc, r=r, deltar=deltar)
		# 	metrix = compute_metrix(v_tr.copy())  # ❗ needs converting to torch, ie tree maps
		# 	wandb.log({'tr/step':step, **metrix})
		# 	print_keys = ['e']
		# 	pprint.pprint(dict(step=step) | {k:v.mean() 
		# 							if isinstance(v, torch.Tensor) else v for k,v in v_tr.items() if k in print_keys})
		
		# if not (step-1):
		# 	print('End Of 1')

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