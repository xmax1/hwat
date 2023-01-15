from functools import reduce
from typing import List

import torch 
import torch.nn as nn

from torch.jit import Final
from functorch import grad, vjp, jvp


def fb_block(s_v: torch.Tensor, p_v: torch.Tensor, n_u: int, n_d: int):
	n_e = n_u + n_d
	sfb_v = [torch.tile(_v.mean(dim=1)[:, None, :], (1, n_e, 1)) for _v in torch.split(s_v, (n_u, n_d), dim=1)] # two element list of (n_batch, n_e, n_sv) tensor
	pfb_v = [_v.mean(dim=1) for _v in torch.split(p_v, (n_u, n_d), dim=1)] # two element list of (n_batch, n_e, n_pv) tensor
	s_v = torch.cat( sfb_v + pfb_v + [s_v,], dim=-1) # (n_batch, n_e, 3n_sv+2n_pv)
	return s_v

### docs:compile-torchscript-model
# - Final[torch.Tensor] not valid type
# - register_buffer way to include tensor constants

class Ansatz_fb(nn.Module):

	n_e: Final[int]                 # number of electrons
	n_u: Final[int]                 # number of up electrons
	n_d: Final[int]                 # number of down electrons
	n_det: Final[int]               # number of determinants
	n_fb: Final[int]                # number of feedforward blocks
	n_pv: Final[int]                # latent dimension for 2-electron
	n_sv: Final[int]                # latent dimension for 1-electron
	n_a: Final[int]                 # nuclei positions
	with_sign: Final[bool]          # return sign of wavefunction
	a: torch.Tensor      

	def __init__(ii, n_e, n_u, n_d, n_det, n_fb, n_pv, n_sv, a: torch.Tensor, with_sign=False):
		super(Ansatz_fb, ii).__init__()
		ii.n_e = n_e                  # number of electrons
		ii.n_u = n_u                  # number of up electrons
		ii.n_d = n_d                  # number of down electrons
		ii.n_det = n_det              # number of determinants
		ii.n_fb = n_fb                # number of feedforward blocks
		ii.n_pv = n_pv                # latent dimension for 2-electron
		ii.n_sv = n_sv                # latent dimension for 1-electron
  
		ii.register_buffer('a', a)  # tensor
		# ii.a = a       
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

	def forward(ii, r: torch.Tensor):
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
		
		# assert s_wd.shape == (n_batch, ii.n_det, ii.n_d, ii.n_d)

		ra_u, ra_d = torch.split(ra, [ii.n_u, ii.n_d], dim=1) # (n_batch, n_u, n_a, 3), (n_batch, n_d, n_a, 3)

		exp_u = torch.linalg.norm(ra_u, dim=-1, keepdim=True) # (n_batch, n_u, n_a, 1)
		exp_d = torch.linalg.norm(ra_d, dim=-1, keepdim=True) # (n_batch, n_d, n_a, 1)

		# assert exp_d.shape == (n_batch, ii.n_d, ii.a.shape[0], 1)

		exp_u = torch.exp(-exp_u)
		exp_d = torch.exp(-exp_d)
		sum_a_exp_u = exp_u.sum(dim=2).unsqueeze(1)
		sum_a_exp_d = exp_d.sum(dim=2).unsqueeze(1)
		orb_u = s_wu * sum_a_exp_u # (n_batch, n_det, n_u, n_u)
		orb_d = s_wd * sum_a_exp_d # (n_batch, n_det, n_d, n_d)

		# assert orb_u.shape == (n_batch, ii.n_det, ii.n_u, ii.n_u)

		log_psi, sgn = logabssumdet(orb_u, orb_d)

		if ii.with_sign:
			return log_psi.squeeze(), sgn.squeeze()
		else:
			return log_psi.squeeze()


def logabssumdet(orb_u: torch.Tensor, orb_d: torch.Tensor = None):
	dtype, device = orb_u.dtype, orb_u.device
	n_batch, n_det, _, _ = orb_u.shape
	orbs = (orb_u, orb_d)

	maxlogdet = torch.zeros((n_batch,), device=device, dtype=dtype) # initialised for sumlogexp trick (for stability)		                                                
	ones = torch.ones((n_batch, n_det,), device=device, dtype=dtype)
	zeros = torch.zeros((n_batch, n_det,), device=device, dtype=dtype)
 
	det_one_e_all = [v.reshape(n_batch, -1) if v.shape[-1] == 1 else ones for v in orbs]   # (n_batch, n_det), (n_batch, n_det)
	det_one_e = det_one_e_all[0] * det_one_e_all[1] # if both cases satisfy n_u or n_d=1, this is the determinant
	
	slogdets = [torch.linalg.slogdet(v) if v.shape[-1]>1 else (ones, zeros) for v in orbs] # two-element list of list of (n_batch, n_det) tensors
	signs = [v[0] for v in slogdets] # two-element list of (n_batch, n_det) tensors
	logdets = [v[1] for v in slogdets] # two-element list of (n_batch, n_det) tensors

	sign_in = signs[0] * signs[1] # (n_batch, n_det)
	logdet = logdets[0] + logdets[1] # (n_batch, n_det)

	if n_det > 1:
		maxlogdet_other, idx = torch.max(logdet, dim=-1) # (n_batch), (n_batch)
		maxlogdet += maxlogdet_other

	det = sign_in * det_one_e * torch.exp(logdet - maxlogdet)	# (n_batch, n_det)

	psi_ish = det.sum(dim=1) # (n_batch)
	sgn_psi = torch.sign(psi_ish) # (n_batch)
	log_psi = torch.log(torch.abs(psi_ish)) + maxlogdet # (n_batch)

	return log_psi, sgn_psi


compute_vv = lambda v_i, v_j: torch.unsqueeze(v_i, axis=-2)-torch.unsqueeze(v_j, axis=-3)

def compute_emb(r, terms, a=None):  
	dtype, device = r.dtype, r.device
	n_e, _ = r.shape
	eye = torch.eye(n_e)[..., None]

	z = []  
	if 'r' in terms:  
		z += [r]  
	if 'r_len' in terms:  
		z += [torch.linalg.norm(r, axis=-1, keepdims=True)]  
	if 'ra' in terms:  
		z += [(r[:, None, :] - a[None, ...]).reshape(n_e, -1)]  
	if 'ra_len' in terms:  
		z += [torch.linalg.norm(r[:, None, :] - a[None, ...], axis=-1)]
	if 'rr' in terms:
		z += [compute_vv(r, r)]
	if 'rr_len' in terms:  # 2nd order derivative of norm is undefined, so use eye
		z += [torch.linalg.norm(compute_vv(r, r)+eye, axis=-1, keepdims=True) * (torch.ones((n_e,n_e,1), device=device, dtype=dtype)-eye)]
	return torch.concatenate(z, axis=-1)

### energy ###

def compute_pe_b(r, a=None, a_z=None):
	dtype, device = r.dtype, r.device
 
	pe_rr = torch.zeros(r.shape[0], dtype=dtype, device=device)
	pe_ra = torch.zeros(r.shape[0], dtype=dtype, device=device)
	pe_aa = torch.zeros(r.shape[0], dtype=dtype, device=device)

	rr = torch.unsqueeze(r, -2) - torch.unsqueeze(r, -3)
	rr_len = torch.linalg.norm(rr, axis=-1)
	pe_rr += torch.tril(1./rr_len, diagonal=-1).sum((-1,-2))

	if not a is None:
		a, a_z = a[None, :, :], a_z[None, None, :]
		ra = torch.unsqueeze(r, -2) - torch.unsqueeze(a, -3)
		ra_len = torch.linalg.norm(ra, axis=-1)
		pe_ra += (a_z/ra_len).sum((-1,-2))

		if len(a_z) > 1:
			aa = torch.unsqueeze(a, -2) - torch.unsqueeze(a, -3)
			aa_len = torch.linalg.norm(aa, axis=-1)
			pe_aa += torch.tril(1./aa_len, diagonal=-1).sum((-1,-2))

	return (pe_rr - pe_ra + pe_aa).squeeze()  


def compute_ke_b(wf: nn.Module, r: torch.Tensor, ke_method='jvp', elements=False):
	dtype, device = r.dtype, r.device

	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim
	r_flat = r.reshape(n_b, n_jvp)
	ones = torch.ones((n_b, ), device=device, dtype=dtype, requires_grad=True)

	def grad_fn(_r: torch.Tensor):
		lp = wf(_r).sum()
		return torch.autograd.grad(lp, _r, create_graph=True)[0]

	def grad_grad_fn(_r: torch.Tensor):
		g = grad_fn(_r)
		ggs = [torch.autograd.grad(g[:, i], _r, grad_outputs=ones, retain_graph=True)[0] for i in range(n_jvp)]
		ggs = torch.stack(ggs, dim=-1)
		return torch.diagonal(ggs, dim1=1, dim2=2)

	r_flat.requires_grad_(True)
	g = grad_fn(r_flat)
	gg = grad_grad_fn(r_flat)
	r_flat.requires_grad_(False)
 
	if elements:
		return g, gg

	e_jvp = gg + g**2
	return -0.5 * e_jvp.sum(-1)


def dep_ke_comp(
    model: nn.Module, 
    r: torch.Tensor,
    ke_method='vjp', 
    elements=False
):
	dtype, device = r.dtype, r.device
	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim
	ones = torch.ones((n_b,), device=device, dtype=dtype)
	r_flat = r.reshape(n_b, n_jvp)
	r_flat = r_flat.requires_grad_(True).contiguous()
 
	def grad_fn(_r: torch.Tensor):
		lp: torch.Tensor = model(_r)
		g = torch.autograd.grad(lp.sum(), _r, create_graph=True)[0]
		return g

	def grad_grad_fn(_r: torch.Tensor):
		g = grad_fn(_r)
		ggs = [torch.autograd.grad(g[:, i], _r, grad_outputs=ones, retain_graph=True)[0] for i in range(n_jvp)]
		ggs = torch.stack(ggs, dim=-1)
		return torch.diagonal(ggs, dim1=1, dim2=2)
	
	g = grad_fn(r_flat)
	gg = grad_grad_fn(r_flat)

	if elements:
		return g, gg

	e_jvp = gg + g**2
	return -0.5 * e_jvp.sum(-1)
	

### sampling ###
def keep_around_points(r, points, l=1.):
	""" points = center of box each particle kept inside. """
	""" l = length side of box """
	r = r - points[None, ...]
	r = r/l
	r = torch.fmod(r, 1.)
	r = r*l
	r = r + points[None, ...]
	return r

def get_center_points(n_e, center: torch.Tensor, _r_cen=None):
	""" from a set of centers, selects in order where each electron will start """
	""" loop concatenate pattern """
	for r_i in range(n_e):
		r_i = center[[r_i % len(center)]]
		_r_cen = r_i if _r_cen is None else torch.concatenate([_r_cen, r_i])
	return torch.Tensor(_r_cen)


def init_r(n_b, n_e, center_points: torch.Tensor, std=0.1):
	""" init r on different gpus with different rngs """
	dtype, device = center_points.dtype, center_points.device
	return torch.Tensor(center_points + torch.randn((n_b,n_e,3), device=device, dtype=dtype)*std)
	# """ loop concatenate pattern """
	# sub_r = [center_points + torch.randn((n_b,n_e,3), device=device, dtype=dtype)*std for i in range(n_device)]
	# return torch.stack(sub_r, dim=0) if len(sub_r)>1 else sub_r[0][None, ...]

	
def sample_b(model: nn.Module, r_0: torch.Tensor, deltar_0: torch.Tensor, n_corr=10):
	""" metropolis hastings sampling with automated step size adjustment """
	device, dtype = r_0.device, r_0.dtype

	deltar_1 = torch.clip(deltar_0 + 0.01*torch.randn([1,], device=device, dtype=dtype), min=0.005, max=0.5)

	p_0 = torch.exp(model(r_0))**2

	acc = []
	for deltar in [deltar_0, deltar_1]:
		
		for _ in torch.arange(n_corr):
			r_1 = r_0 + torch.randn_like(r_0, device=device, dtype=dtype)*deltar
			p_1 = torch.exp(model(r_1))**2

			p_mask = (p_1/p_0) > torch.rand_like(p_1, device=device, dtype=dtype)
			p_0 = torch.where(p_mask, p_1, p_0)

			r_0 = torch.where(p_mask[..., None, None], r_1, r_0)

		acc += [p_mask.type_as(r_0).mean()]

	mask = ((0.5-acc[0])**2 - (0.5-acc[1])**2) < 0.
	deltar = mask*deltar_0 + ~mask*deltar_1
	
	return r_0, (acc[0]+acc[1])/2., deltar
