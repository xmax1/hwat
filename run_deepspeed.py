#!/usr/bin/env python
import torch

from pyfig import Pyfig
import numpy as np
import pprint
import deepspeed
from torch import nn
from typing import Final
from hwat_b import fb_block, logabssumdet, keep_around_points, compute_ke_b, compute_pe_b

import wandb
from utils import compute_metrix
from hwat_b import init_r, get_center_points


# 


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

	def __init__(ii, 
            n_e, n_u, n_d, n_det, n_fb, n_pv, n_sv, 
            a: torch.Tensor, 
            r: torch.Tensor, 
            deltar: torch.Tensor,
            center_points: torch.Tensor,
            with_sign=False
        ):
		super(ModelSample, ii).__init__()
		ii.device = a.device
		ii.dtype = a.dtype
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
		ii.n_corr = c.data.n_corr

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

		ii.r = r.to(ii.device).to(ii.dtype)
		ii.deltar = deltar.to(ii.device).to(ii.dtype)
		ii.center_points = center_points.to(ii.device).to(ii.dtype)
		ii.acc = 0.
		ii.step = 1

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
		
	def compute_grads(ii):
		with torch.no_grad():
			ii.r = ii.sample()  # ❗needs testing 
			ii.r = keep_around_points(ii.r, ii.center_points, l=5.) if ii.step < 50 else ii.r
		
		ii.zero_grad()
		ii.ke = compute_ke_b(ii.wf, ii.r, ke_method=c.model.ke_method)
		ii.pe = compute_pe_b(ii.r, c.data.a, c.data.a_z)
		e = ii.pe + ii.ke
		e_mean_dist = torch.mean(torch.abs(torch.median(e) - e))
		e_clip = torch.clip(e, min=e-5*e_mean_dist, max=e+5*e_mean_dist)
		ii.e = e
		ii.step += 1
		return (e_clip - e_clip.mean()).detach()

	def forward(ii):
		energy_term = ii.compute_grads()
		ii.zero_grad()
		return (energy_term * ii.wf(ii.r)).mean()

	def v_tr(ii):
		return dict(
			r = ii.r.detach().cpu().numpy(), 
            params=[p.detach() for p in ii.parameters()],
			e = ii.e.detach().cpu().numpy(),
			pe = ii.pe.detach().cpu().numpy(),
			ke = ii.ke.detach().cpu().numpy(),
    )

	def sample(ii):
		""" metropolis hastings sampling with automated step size adjustment """

		deltar_1 = torch.clip(ii.deltar + 0.01*torch.randn([1,], device=ii.device, dtype=ii.dtype), min=0.005, max=0.5)
		r_0 = ii.r
		p_0 = torch.exp(ii.wf(r_0))**2  			# ❗can make more efficient with where modelment at end

		acc = []
		for deltar in [ii.deltar, deltar_1]:
			
			for _ in torch.arange(ii.n_corr):
				
				r_1 = r_0 + torch.randn_like(r_0, device=ii.device, dtype=ii.dtype)*deltar

				p_1 = torch.exp(ii.wf(r_1))**2
	
				p_mask = (p_1/p_0) > torch.rand_like(p_1, device=ii.device, dtype=ii.dtype)		# metropolis hastings

				r_0 = torch.where(p_mask[..., None, None], r_1, r_0)
				p_0 = torch.where(p_mask, p_1, p_0)

			acc += [p_mask.type_as(r_0).mean()]

		mask = ((0.5-acc[0])**2 - (0.5-acc[1])**2) < 0.
		ii.deltar = mask*ii.deltar + ~mask*deltar_1
		return r_0

def run(c: Pyfig):
	deepspeed.init_distributed()

	torch.manual_seed(1234)
	torch.set_default_tensor_type(torch.FloatTensor)   # ❗ Ensure works when default not set AND can go float32 or 64
 
	n_device = c.n_device
	print(f'🤖 {n_device} GPUs available')

	_dummy = torch.randn((1,))
	dtype = _dummy.dtype
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	c._convert(device=device, dtype=dtype)

	center_points = get_center_points(c.data.n_e, c.data.a).to(device).to(dtype)
	r = init_r(c.data.n_b, c.data.n_e, center_points, std=0.1).to(device).to(dtype)
	deltar = torch.tensor([0.02]).to(device).to(dtype)
 
	model_sample = c.partial(ModelSample, 
        a=c.data.a, r=r, deltar=deltar, center_points=center_points
    ).to(dtype).to(device)

	print(f"""exp/actual | 
		cps    : {(c.data.n_e, 3)}/{center_points.shape}
		r      : {(c.n_device, c.data.n_b, c.data.n_e, 3)}/{r.shape}
		deltar : {(c.n_device, 1)}/{deltar.shape}
	""")
 
	model_engine, opt, _, _ = \
		deepspeed.initialize(
      args=None, 
      model=model_sample, 
      model_parameters=model_sample.parameters(), 
      config=c.ds_c.path,
    )

	model_sample()

	wandb.define_metric("*", step_metric="tr/step")
	for step in range(1, c.n_step+1):
		
		loss = model_engine()
		model_engine.backward(loss)
		model_engine.step()

		print('Loss: ', loss.mean())

		if not (step % c.log_metric_step):
			try:
				v_tr = model_engine.v_tr().copy()

				metrix = compute_metrix(v_tr)  # ❗ needs converting to torch, ie tree maps
				wandb.log({'tr/step':step, **metrix})
				print_keys = ['e']
				pprint.pprint(dict(step=step) | {k:v.mean() 
					if isinstance(v, torch.Tensor) else v for k,v in v_tr.items() if k in print_keys})
			except Exception as e:
				print(e)

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

	# import json
	# ds_config = {
	# 	"train_batch_size" : global_batch_size,
	# 	"train_micro_batch_size_per_gpu": micro_batch_size,
	# 	"steps_per_print": 1,
	# 	"gradient_accumulation_steps": 1,
	# 	"zero_optimization": {
	# 	"stage": 3,
	# 	"stage3_max_live_parameters": 3e9,
	# 	"stage3_max_reuse_distance": 3e9,
	# 	"stage3_param_persistence_threshold": 1e5,
	# 	"stage3_prefetch_bucket_size": 5e7,
	# 	"contiguous_gradients": True,
	# 	"overlap_comm": True,
	# 	"reduce_bucket_size": 90000000,
	# 	"sub_group_size": 1e9,
	# 	"offload_optimizer": {
	# 		"device": "none",
	# 		"buffer_count": 4,
	# 		"pipeline_read": False,
	# 		"pipeline_write": False,
	# 		"pin_memory": True
	# 	}
	# 	},
	# 	"gradient_clipping": 1.0,
	# 	"fp16": {
	# 	"enabled": True,
	# 	"initial_scale_power" : 15,
	# 	"loss_scale_window": 1000,
	# 	"hysteresis": 2,
	# 	"min_loss_scale": 1
	# 	},
	# 	"wall_clock_breakdown": True,
	# 	"zero_allow_untested_optimizer": False,
	# 	"aio": {
	# 	"block_size": 1048576,
	# 	"queue_depth": 16,
	# 	"single_submit": False,
	# 	"overlap_events": True,
	# 	"thread_count": 2
	# 	}
	# }

	
# // {
# //   "train_batch_size" : 64,
# //   "optimizer": {
# //     "type": "Adam",
# //     "params": {
# //       "lr": 0.0002,
# //       "betas": [
# //         0.5,
# //         0.999
# //       ],
# //       "eps": 1e-8
# //     }
# //   },
# //   "steps_per_print" : 10
# // }


	# # Place ds_config.json in the same folder as pretrain_gpt.py (script to run)
	# ds_config_path = '../../ds_config.json'
	# with open(ds_config_path, 'w') as fp:
	# 	json.dump(ds_config, fp, indent=4)
  
	run(c)
 
 
	# import re
	# check deepspeed installation
	# report = !python3 -m deepspeed.env_report
	# r = re.compile('.*ninja.*OKAY.*')
	# assert any(r.match(line) for line in report) == True, "DeepSpeed Inference not correct installed"

	# # check cuda and torch version
	# torch_version, cuda_version = torch.__version__.split("+")
	# torch_version = ".".join(torch_version.split(".")[:2])
	# cuda_version = f"{cuda_version[2:4]}.{cuda_version[4:]}"
	# r = re.compile(f'.*torch.*{torch_version}.*')
	# assert any(r.match(line) for line in report) == True, "Wrong Torch version"
	# r = re.compile(f'.*cuda.*{cuda_version}.*')
	# assert any(r.match(line) for line in report) == True, "Wrong Cuda version"