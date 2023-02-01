import torch 
from torch import Tensor
import torch.nn as nn

from torch.jit import Final
from functorch import grad
import functorch
import pprint
from things.utils import debug_dict
from copy import deepcopy
import numpy as np
from typing import Callable, Any


def fb_block(s_v: Tensor, p_v: Tensor, n_u: int, n_d: int):
	n_e = n_u + n_d

	fb_f = [s_v,]
	for s_x in torch.split(s_v, (n_u, n_d), dim=1):
		mean_s_x = s_x.mean(dim=1, keepdim=True).tile((1, n_e, 1))
		fb_f += [mean_s_x] # two element list of (n_b, n_e, n_sv) tensor
	
	for p_x in torch.split(p_v, (n_u, n_d), dim=1):
		mean_p_x = p_x.mean(dim=1)
		fb_f += [mean_p_x] # two element list of (n_b, n_e, n_sv) tensor

	return torch.cat(fb_f, dim=-1) # (n_b, n_e, 3n_sv+2n_pv)


class Ansatz_fb(nn.Module):

	n_d: Final[int]                 	# number of down electrons
	n_det: Final[int]               	# number of determinants
	n_fb: Final[int]                	# number of feedforward blocks
	n_pv: Final[int]                	# latent dimension for 2-electron
	n_sv: Final[int]                	# latent dimension for 1-electron
	n_a: Final[int]                 	# nuclei positions
	with_sign: Final[bool]          	# return sign of wavefunction
	debug: Final[bool]          		# return sign of wavefunction
	a: Tensor      

	def __init__(
		ii, 
		n_e, 
		n_u, 
		n_d, 
		n_det, 
		n_fb, 
		n_pv, 
		n_sv, 
		a, 
		*, 
		with_sign=False):
		super(Ansatz_fb, ii).__init__()
  
		ii.with_sign = with_sign      # return sign of wavefunction

		ii.n_e: Final[int] = n_e      					# number of electrons
		ii.n_u: Final[int] = n_u                  # number of up electrons
		ii.n_d: Final[int] = n_d                  # number of down electrons
		ii.n_det = n_det              # number of determinants
		ii.n_fb = n_fb                # number of feedforward blocks
		ii.n_pv = n_pv                # latent dimension for 2-electron
		ii.n_sv = n_sv                # latent dimension for 1-electron
		ii.n_sv_out = 3*n_sv+2*n_pv

		ii.n_a = len(a)

		n_p_in = 4
		n_s_in = 4*ii.n_a
	
		s_size = [(n_s_in*3 + n_p_in*2, n_sv),] + [(ii.n_sv_out, n_sv),]*n_fb
		p_size = [(n_p_in, n_pv),] + [(n_pv, n_pv),]*n_fb
		print('model fb layers: \n', s_size, p_size)

		ii.s_lay = nn.ModuleList([nn.Linear(*dim) for dim in s_size])
		ii.p_lay = nn.ModuleList([nn.Linear(*dim) for dim in p_size])

		ii.v_after = nn.Linear(ii.n_sv_out, ii.n_sv)

		ii.w_spin_u = nn.Linear(ii.n_sv, ii.n_u * ii.n_det)
		ii.w_spin_d = nn.Linear(ii.n_sv, ii.n_d * ii.n_det)
		ii.w_spin = [ii.w_spin_u, ii.w_spin_d]

		ii.w_final = nn.Linear(ii.n_det, 1, bias=False)

		ii.register_buffer('a', a.detach().clone().requires_grad_(False), persistent=False)    # tensor
		ii.register_buffer('eye', torch.eye(ii.n_e)[None, :, :, None].requires_grad_(False), persistent=False)
		ii.register_buffer('ones', torch.ones((1, ii.n_det)).requires_grad_(False), persistent=False)
		ii.register_buffer('zeros', torch.zeros((1, ii.n_det)).requires_grad_(False), persistent=False)
  
		debug_dict(msg='model buffers', buffers=list(ii.buffers()))


	def forward(ii, r: Tensor) -> Tensor:
		
		if r.ndim in (1, 2):
			r = r.reshape(-1, ii.n_e, 3) # (n_b, n_e, 3)
		
		n_b = r.shape[0]

		ra = r[:, :, None, :] - ii.a[:, :] # (n_b, n_e, n_a, 3)
		ra_u, ra_d = torch.split(ra, [ii.n_u, ii.n_d], dim=1) # (n_b, n_u(r), n_a, 3), (n_b, n_d(r), n_a, 3)

		s_v, p_v = ii.compute_embedding(r, ra) # (n_b, n_e, n_sv), (n_b, n_e, n_pv)
		s_v = ii.compute_stream(s_v, p_v) # (n_b, n_e, n_sv), (n_b, n_e, n_sv)
		
		s_u, s_d = torch.split(s_v, [ii.n_u, ii.n_d], dim=1) # (n_b, n_u, 3n_sv+2n_pv), (n_b, n_d, 3n_sv+2n_pv)

		### case with 1 electron will fail ### # e1mask = torch.sign(orb_u.shape[-1])
		orb_u = ii.compute_orb_from_stream(s_u, ra_u, spin=0) # (n_b, n_det, n_u, n_u)
		orb_d = ii.compute_orb_from_stream(s_d, ra_d, spin=1) # (n_b, n_det, n_d, n_d)

		sign_u, logdet_u = torch.linalg.slogdet(orb_u)
		sign_d, logdet_d = torch.linalg.slogdet(orb_d)

		sign = sign_u * sign_d
		logdet = logdet_u + logdet_d

		maxlogdet, _ = torch.max(logdet, dim=-1, keepdim=True) # (n_b, 1), (n_b, 1)
		
		sub_det = sign * torch.exp(logdet - maxlogdet) 
		
		psi_ish: Tensor = ii.w_final(sub_det)  # (n_b, 1)

		sign_psi = torch.sign(psi_ish)  

		log_psi = torch.log(torch.absolute(psi_ish.squeeze())) + maxlogdet.squeeze()
  
		if ii.with_sign:
			return log_psi, sign_psi

		return log_psi

	def compute_orb(ii, r: Tensor) -> tuple[Tensor, Tensor]:
		ra = r[:, :, None, :] - ii.a[:, :] # (n_b, n_e, n_a, 3)
		ra_u, ra_d = torch.split(ra, [ii.n_u, ii.n_d], dim=1) # (n_b, n_u(r), n_a, 3), (n_b, n_d(r), n_a, 3)
		s_v, p_v = ii.compute_embedding(r, ra)
		s_v = ii.compute_stream(s_v, p_v) # (n_b, n_u, n_det), (n_b, n_d, n_det)
		s_u, s_d = torch.split(s_v, [ii.n_u, ii.n_d], dim=1) # (n_b, n_u, 3n_sv+2n_pv), (n_b, n_d, 3n_sv+2n_pv)
		orb_u = ii.compute_orb_from_stream(s_u, ra_u, spin=0) # (n_b, n_u, n_u, n_det)
		orb_d = ii.compute_orb_from_stream(s_d, ra_d, spin=1) # (n_b, n_d, n_d, n_det)
		return orb_u, orb_d

	def compute_embedding(ii, r: Tensor, ra: Tensor) -> tuple[Tensor, Tensor]:
		n_b, n_e, _ = r.shape

		ra_len = torch.linalg.vector_norm(ra, dim=-1, keepdim=True) # (n_b, n_e, n_a, 1)

		rr = r[:, None, :, :] - r[:, :, None, :] # (n_b, n_e, n_e, 1)
		rr_len = torch.linalg.vector_norm(rr + ii.eye, dim=-1, keepdim=True) 

		s_v = torch.cat([ra, ra_len], dim=-1).reshape(n_b, ii.n_e, -1) # (n_b, n_e, n_a*4)
		p_v = torch.cat([rr, rr_len], dim=-1) # (n_b, n_e, n_e, 4)

		return s_v, p_v

	def compute_stream(ii, s_v: Tensor, p_v: Tensor):
			
		s_v_block = fb_block(s_v, p_v, ii.n_u, ii.n_d)

		for l_i, (s_lay, p_lay) in enumerate(zip(ii.s_lay, ii.p_lay)):

			s_v_tmp = torch.tanh(s_lay(s_v_block))
			s_v = s_v_tmp + s_v if l_i else s_v_tmp

			p_v_tmp = torch.tanh(p_lay(p_v))
			p_v = p_v_tmp + p_v if l_i else p_v_tmp

			s_v_block = fb_block(s_v, p_v, ii.n_u, ii.n_d) # (n_b, n_e, 3n_sv+2n_pv)

		s_v: Tensor = torch.tanh(ii.v_after(s_v_block)) # (n_b, n_u(r), n_sv)    # n_sv//2)
		return s_v

	def compute_orb_from_stream(ii, s_v: Tensor, ra_v: Tensor, spin: int) -> Tensor:
		n_b, n_spin, _ = s_v.shape

		s_w = ii.w_spin[spin](s_v).reshape(n_b, n_spin, n_spin, ii.n_det) # (n_b, n_det, n_d, n_d)

		exp = torch.exp(-torch.linalg.vector_norm(ra_v, dim=-1)) # (n_b, n_spin(r), n_a)
		sum_a_exp = exp.sum(dim=-1, keepdim=True)[..., None] # n_b, n_u(r)
		
		orb = s_w * sum_a_exp # (n_b, n_u(r), n_u(orb), n_det) 
		return orb.transpose(-1, 1) # (n_b, n_det, n_u(r), n_u(orb))

	def compute_hf_orb(ii, r: Tensor, mol: Any, hf: Any):
		n_b, n_e, _ = r.shape
		r_hf = r.reshape(-1, 3) # (n_b*n_e, 3)
		r_hf = r_hf.detach().cpu().numpy()
		ao = mol.eval_gto('GTOval', r_hf) # (n_b*n_e, n_ao)
		ao = ao.reshape(n_b, n_e, -1) # (n_b, n_e, n_ao)
		mo_coef = torch.tensor(hf.mo_coeff).to(device=r.device, dtype=r.dtype).requires_grad_(True) # (n_b, n_e, n_mo)
		ao = torch.tensor(ao).to(device=r.device, dtype=r.dtype).requires_grad_(True) # (n_b, n_e, n_mo)
		mo = torch.einsum("bea,sao->sbeo", ao, mo_coef)
		mo = mo.unsqueeze(2).tile((1, 1, ii.n_det, 1, 1))
		return mo[0][..., :ii.n_u, :ii.n_u], mo[1][..., :ii.n_d, :ii.n_d]


	def full_det_from_spin_det(ii, orb_u: Tensor, orb_d: Tensor):
		device, dtype = orb_u.device, orb_u.dtype
		n_b, n_det, n_u, n_u = orb_u.shape
		_, _, n_d, n_d = orb_d.shape
		zero_top = torch.zeros((n_b, n_det, n_u, n_d), device=device, dtype=dtype)
		zero_bot = torch.transpose(zero_top, -1, -2)
		orb_u = torch.cat([orb_u, zero_top], dim=-1)
		orb_d = torch.cat([zero_bot, orb_d], dim=-1)
		return torch.cat([orb_u, orb_d], dim=-2)
		

compute_vv = lambda v_i, v_j: torch.unsqueeze(v_i, dim=-2)-torch.unsqueeze(v_j, dim=-3)

def compute_emb(r: Tensor, terms: list, a=None):  
	dtype, device = r.dtype, r.device
	n_e, _ = r.shape
	eye = torch.eye(n_e)[..., None]

	z = []  
	if 'r' in terms:  
		z += [r]  
	if 'r_len' in terms:  
		z += [torch.linalg.norm(r, dim=-1, keepdims=True)]  
	if 'ra' in terms:  
		z += [(r[:, None, :] - a[None, ...]).reshape(n_e, -1)]  
	if 'ra_len' in terms:  
		z += [torch.linalg.norm(r[:, None, :] - a[None, ...], dim=-1)]
	if 'rr' in terms:
		z += [compute_vv(r, r)]
	if 'rr_len' in terms:  # 2nd order derivative of norm is undefined, so use eye
		z += [torch.linalg.norm(compute_vv(r, r)+eye, dim=-1, keepdims=True)]
	return torch.concatenate(z, dim=-1)

### energy ###

def compute_pe_b(r, a=None, a_z=None):

	rr = torch.unsqueeze(r, -2) - torch.unsqueeze(r, -3)
	rr_len = torch.linalg.norm(rr, dim=-1)
	pe = torch.tril(1./rr_len, diagonal=-1).sum((-1,-2))

	if not a is None:
		a, a_z = a[None, :, :], a_z[None, None, :]
		ra = torch.unsqueeze(r, -2) - torch.unsqueeze(a, -3)
		ra_len = torch.linalg.norm(ra, dim=-1)
		pe -= (a_z/ra_len).sum((-1,-2))

		if len(a_z) > 1:
			aa = torch.unsqueeze(a, -2) - torch.unsqueeze(a, -3)
			aa_len = torch.linalg.norm(aa, dim=-1)
			pe += torch.tril(1./aa_len, diagonal=-1).sum((-1,-2))

	return pe.squeeze()  



# torch:issue !!! torch.no_grad does not work with torch.autograd.grad 

def compute_ke_b(
	model: nn.Module, 
	model_fn: Callable,
	r: Tensor,
	ke_method='vjp', 
	elements=False
):
	dtype, device = r.dtype, r.device
	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim

	params = [p.detach() for p in model.parameters()]
	buffers = [b for b in model.buffers()]
	model_rv = lambda _r: model_fn(params, buffers, _r).sum()

	ones = torch.ones((n_b,), device=device, dtype=dtype)
	eyes = torch.eye(n_jvp, dtype=dtype, device=device)[None].repeat((n_b, 1, 1))

	r_flat = r.reshape(n_b, n_jvp).detach().contiguous().requires_grad_(True)

	assert r_flat.requires_grad
	for p in params:
		assert not p.requires_grad

	def ke_grad_grad_method(r_flat):

		def grad_fn(_r: Tensor):
			lp: Tensor = model(_r)
			g = torch.autograd.grad(lp.sum(), _r, create_graph=True)[0]
			return g

		def grad_grad_fn(_r: Tensor):
			g = grad_fn(_r)
			ggs = [torch.autograd.grad(g[:, i], _r, grad_outputs=ones, retain_graph=True)[0] for i in range(n_jvp)]
			ggs = torch.stack(ggs, dim=-1)
			return torch.diagonal(ggs, dim1=1, dim2=2)

		g = grad_fn(r_flat)
		gg = grad_grad_fn(r_flat)
		return g, gg

	def ke_vjp_method(r_flat):
		grad_fn = functorch.grad(model_rv)
		g, fn = functorch.vjp(grad_fn, r_flat)
		gg = torch.stack([fn(eyes[..., i])[0][:, i] for i in range(n_jvp)], dim=-1)
		return g, gg

	def ke_jvp_method(r_flat):
		grad_fn = functorch.grad(model_rv)
		jvp_all = [functorch.jvp(grad_fn, (r_flat,), (eyes[:, i],)) for i in range(n_jvp)]  # (grad out, jvp)
		g = torch.stack([x[:, i] for i, (x, _) in enumerate(jvp_all)], dim=-1)
		gg = torch.stack([x[:, i] for i, (_, x) in enumerate(jvp_all)], dim=-1)
		return g, gg

	ke_function = dict(
		grad_grad	= ke_grad_grad_method, 
		vjp			= ke_vjp_method, 
		jvp			= ke_jvp_method
	)[ke_method]

	g, gg = ke_function(r_flat) 

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


def get_center_points(n_e, center: np.ndarray|Tensor, _r_cen=None):
	""" from a set of centers, selects in order where each electron will start """
	""" loop concatenate pattern """
	for r_i in range(n_e):
		r_i = center[[r_i % len(center)]]
		_r_cen = r_i if _r_cen is None else torch.concatenate([_r_cen, r_i])
	return _r_cen


def init_r(center_points: Tensor, std=0.1):
	""" init r on different gpus with different rngs """
	return center_points + torch.randn_like(center_points)*std
	# """ loop concatenate pattern """
	# sub_r = [center_points + torch.randn((n_b,n_e,3), device=device, dtype=dtype)*std for i in range(n_device)]
	# return torch.stack(sub_r, dim=0) if len(sub_r)>1 else sub_r[0][None, ...]


def is_a_larger(a: Tensor, b: Tensor) -> Tensor:
	""" given a condition a > b returns 1 if true and 0 if not """
	v = a - b # +ve when a bigger
	a_is_larger = (torch.sign(v) + 1.) / 2. # 1 when a bigger
	b_is_larger = (a_is_larger-1.)*-1. # 1 when b bigger
	return a_is_larger, b_is_larger

@torch.no_grad()
def sample_b(
	model: nn.Module, 
	data: Tensor, 
	deltar: Tensor, 
	n_corr: int=10
):
	""" metropolis hastings sampling with automated step size adjustment """
	device, dtype = data.device, data.dtype

	p_0 = torch.exp(model(data))**2

	deltar_1 = torch.clip(deltar + 0.01*torch.randn_like(deltar), min=0.005, max=0.5)
	
	acc = torch.zeros_like(deltar)
	acc_all = torch.zeros_like(deltar)

	for dr_test in [deltar, deltar_1]:
		for _ in torch.arange(1, n_corr+1):
			
			data_1 = data + torch.randn_like(data, device=device, dtype=dtype, layout=torch.strided)*dr_test
			p_1 = torch.exp(model(data_1))**2
			

			alpha = torch.rand_like(p_1, device=device, dtype=dtype, layout=torch.strided)
			a_larger, b_larger = is_a_larger(p_1/p_0, alpha)

			p_0 = a_larger*p_1 + b_larger*p_0
			data = a_larger.unsqueeze(-1).unsqueeze(-1)*data_1 + b_larger.unsqueeze(-1).unsqueeze(-1)*data

			acc_test = torch.mean(a_larger, dim=0, keepdim=True) # docs:torch:knowledge keepdim requires dim=int|tuple[int]

		del data_1
		del p_1
		del a_larger
		del b_larger

		acc_all += acc_test

		a_larger, b_larger = is_a_larger(torch.absolute(0.5-acc), torch.absolute(0.5-acc_test))

		acc = a_larger*acc_test + b_larger*acc

		deltar = a_larger*dr_test + b_larger*deltar
	
	return dict(data=data, acc=acc_all/2., deltar=deltar)


from torch.utils.data import Dataset
from pyfig import Pyfig


class PyfigDataset(Dataset):

	def __init__(ii, 
		c: Pyfig,
		model: nn.Module,
		init_scale: float=0.1
	):

		ii.model = model
		ii.n_b = c.data.n_b

		ii.n_step = c.n_step
		ii.n_corr = c.app.n_corr
		ii.deltar = torch.tensor([0.02,], device=c.device, dtype=c.dtype).requires_grad_(False)
		ii.center_points = get_center_points(c.app.n_e, c.app.a)

		ii.data = ii.center_points \
			+ init_scale * torch.randn(size=(c.data.n_b, *ii.center_points.shape)).to(c.device, c.dtype)

		print('dataset:init data', ii.data.shape, 'deltar', ii.deltar.shape, 'center_points', ii.center_points.shape)

	def __len__(ii):
		return ii.n_b * ii.n_step

	def __getitem__(ii, i):

		v_d = sample_b(ii.model, ii.data, ii.deltar, n_corr=ii.n_corr)

		for k,v in v_d.items():
			setattr(ii, k, v)

		return ii.data