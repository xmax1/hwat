from functools import reduce, partial
from typing import Any
from utils import wpr
from typing import Callable
import functools

### model ###
from functools import reduce
import torch 
import torch.nn as nn

def logabssumdet(xs):
		
		dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]						# in case n_u or n_d=1, no need to compute determinant
		dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1.					# take product of these cases
		maxlogdet = 0.																# initialised for sumlogexp trick (for stability)
		det = dets																	# if both cases satisfy n_u or n_d=1, this is the determinant
		
		slogdets = [torch.linalg.slogdet(x) for x in xs if x.shape[-1]>1] 			# otherwise take slogdet
		if len(slogdets)>0: 
			sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)  # take product of n_u or n_d!=1 cases
			maxlogdet = torch.max(logdet)											# adjusted for new inputs
			det = sign_in * dets * torch.exp(logdet-maxlogdet)						# product of all these things is determinant
		
		psi_ish = torch.sum(det)
		sgn_psi = torch.sign(psi_ish)
		log_psi = torch.log(torch.abs(psi_ish)) + maxlogdet
		return log_psi, sgn_psi


class FermiNetTorch(nn.Module):
	def __init__(self, n_e, n_u, n_d, n_det, n_fb, n_pv, n_sv, a, with_sign=False):
		super(FermiNetTorch, self).__init__()
		self.n_e = n_e                  # number of electrons
		self.n_u = n_u                  # number of up electrons
		self.n_d = n_d                  # number of down electrons
		self.n_det = n_det              # number of determinants
		self.n_fb = n_fb                # number of feedforward blocks
		self.n_pv = n_pv                # latent dimension for 2-electron
		self.n_sv = n_sv                # latent dimension for 1-electron
		self.a = a                      # nuclei positions
		self.with_sign = with_sign      # return sign of wavefunction

		self.n1 = [4*self.a.shape[0]] + [self.n_sv]*self.n_fb
		self.n2 = [4] + [self.n_pv]*(self.n_fb - 1)
		assert (len(self.n1) == self.n_fb+1) and (len(self.n2) == self.n_fb)
		self.Vs = nn.ModuleList([nn.Linear(3*self.n1[i]+2*self.n2[i], self.n1[i+1]) for i in range(self.n_fb)])
		self.Ws = nn.ModuleList([nn.Linear(self.n2[i], self.n2[i+1]) for i in range(self.n_fb-1)])

		self.V_half_u = nn.Linear(self.n_sv, self.n_sv // 2)
		self.V_half_d = nn.Linear(self.n_sv, self.n_sv // 2)

		self.wu = nn.Linear(self.n_sv // 2, self.n_u)
		self.wd = nn.Linear(self.n_sv // 2, self.n_d)

		# TODO: Multideterminant. If n_det > 1 we should map to n_det*n_u (and n_det*n_d) instead,
		#  and then split these outputs in chunks of n_u (n_d)
		# TODO: implement layers for sigma and pi

	def forward(self, r: torch.Tensor):
		"""
		Batch dimension is not yet supported.
		"""

		if len(r.shape) == 1:
			r = r.reshape(self.n_e, 3) # (n_e, 3)

		eye = torch.eye(self.n_e, device=r.device).unsqueeze(-1)

		ra = r[:, None, :] - self.a[None, :, :] # (n_e, n_a, 3)
		ra_len = torch.norm(ra, dim=-1, keepdim=True) # (n_e, n_a, 1)

		rr = r[None, :, :] - r[:, None, :] # (n_e, n_e, 1)
		rr_len = torch.norm(rr+eye, dim=-1, keepdim=True) * (torch.ones((self.n_e, self.n_e, 1))-eye) # (n_e, n_e, 1) 
		# TODO: Just remove '+eye' from above, it's unnecessary

		s_v = torch.cat([ra, ra_len], dim=-1).reshape(self.n_e, -1) # (n_e, n_a*4)
		p_v = torch.cat([rr, rr_len], dim=-1) # (n_e, n_e, 4)

		for l, (V, W) in enumerate(zip(self.Vs, self.Ws)):
			sfb_v = [torch.tile(_v.mean(dim=0)[None, :], (self.n_e, 1)) for _v in torch.split(s_v, 2, dim=0)]
			pfb_v = [_v.mean(dim=0) for _v in torch.split(p_v, self.n_u, dim=0)]
			
			s_v = torch.cat(sfb_v+pfb_v+[s_v,], dim=-1) # s_v = torch.cat((s_v, sfb_v[0], sfb_v[1], pfb_v[0], pfb_v[0]), dim=-1)
			s_v = torch.tanh(V(s_v)) + (s_v if (s_v.shape[-1]==self.n_sv) else 0.)
			
			if not (l == (self.n_fb-1)):
				p_v = torch.tanh(W(p_v)) + (p_v if (p_v.shape[-1]==self.n_pv) else 0.)
		
		s_u, s_d = torch.split(s_v, self.n_u, dim=0)

		s_u = torch.tanh(self.V_half_u(s_u)) # spin dependent size reduction
		s_d = torch.tanh(self.V_half_d(s_d))

		s_wu = self.wu(s_u) # map to phi orbitals
		s_wd = self.wd(s_d)

		assert s_wd.shape == (self.n_d, self.n_d)

		ra_u, ra_d = torch.split(ra, self.n_u, dim=0)

		# TODO: implement sigma = nn.Linear() before this
		exp_u = torch.norm(ra_u, dim=-1, keepdim=True)
		exp_d = torch.norm(ra_d, dim=-1, keepdim=True)

		assert exp_d.shape == (self.n_d, self.a.shape[0], 1)

		# TODO: implement pi = nn.Linear() before this
		orb_u = (s_wu * (torch.exp(-exp_u).sum(axis=1)))[None, :, :]
		orb_d = (s_wd * (torch.exp(-exp_d).sum(axis=1)))[None, :, :]

		assert orb_u.shape == (1, self.n_u, self.n_u)

		log_psi, sgn = logabssumdet([orb_u, orb_d])

		if self.with_sign:
			return log_psi, sgn
		else:
			return log_psi.squeeze()
		
compute_vv = lambda v_i, v_j: torch.unsqueeze(v_i, axis=-2)-torch.unsqueeze(v_j, axis=-3)

def compute_emb(r, terms, a=None):  
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
		z += [torch.linalg.norm(compute_vv(r, r)+eye, axis=-1, keepdims=True) * (torch.ones((n_e,n_e,1))-eye)]
	return torch.concatenate(z, axis=-1)

def logabssumdet(xs):
	
	dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]						# in case n_u or n_d=1, no need to compute determinant
	dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1.					# take product of these cases
	maxlogdet = 0.																# initialised for sumlogexp trick (for stability)
	det = dets																	# if both cases satisfy n_u or n_d=1, this is the determinant
	
	slogdets = [torch.linalg.slogdet(x) for x in xs if x.shape[-1]>1] 			# otherwise take slogdet
	if len(slogdets)>0: 
		sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)  # take product of n_u or n_d!=1 cases
		maxlogdet = torch.max(logdet)												# adjusted for new inputs
		det = sign_in * dets * torch.exp(logdet-maxlogdet)						# product of all these things is determinant
	
	psi_ish = torch.sum(det)
	sgn_psi = torch.sign(psi_ish)
	log_psi = torch.log(torch.abs(psi_ish)) + maxlogdet
	return log_psi, sgn_psi

### energy ###

def compute_pe_b(r, a=None, a_z=None):
	rr = torch.unsqueeze(r, -2) - torch.unsqueeze(r, -3)
	rr_len = torch.linalg.norm(rr, axis=-1)
	pe_rr = torch.tril(1./rr_len, k=-1).sum((1,2))

	if a is None:
		return pe_rr
	
	a, a_z = a[None, :, :], a_z[None, None, :]
	ra = torch.unsqueeze(r, -2) - torch.unsqueeze(a, -3)
	ra_len = torch.linalg.norm(ra, axis=-1)
	pe_ra = (a_z/ra_len).sum((1,2))   

	if len(a) > 1:  # len(a) = n_a
		aa = torch.unsqueeze(a, -2) - torch.unsqueeze(a, -3)
		aa_len = torch.linalg.norm(aa, axis=-1)
		pe_aa = torch.tril(1./aa_len, k=-1).sum((1,2))
	return (pe_rr - pe_ra + pe_aa).squeeze()


def compute_ke_b(state, r):
	
	grads = torch.autograd.grad(lambda r: state(r).sum(), r, create_graph=True)
	
	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim
	r = r.reshape(n_b, n_jvp)
	eye = torch.eye(n_jvp, dtype=r.dtype)[None, ...].repeat(n_b, axis=0)
	
	def _body_fun(i, val):
		primal, tangent = jax.jvp(grad_fn, (r,), (eye[..., i],))  
		return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	
	return (- 0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, torch.zeros(n_b,))).squeeze()

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

def get_center_points(n_e, center, _r_cen=None):
	""" from a set of centers, selects in order where each electron will start """
	""" loop concatenate pattern """
	for r_i in range(n_e):
		_r_cen = center[[r_i % len(center)]] if _r_cen is None else torch.concatenate([_r_cen, center[[r_i % len(center)]]])
	return _r_cen

def init_r(rng, n_b, n_e, center_points, std=0.1):
	""" init r on different gpus with different rngs """
	""" loop concatenate pattern """
	sub_r = [center_points + rnd.normal(rng_i,(n_b,n_e,3))*std for rng_i in rng]
	return torch.stack(sub_r) if len(sub_r)>1 else sub_r[0][None, ...]

def sample_b(rng, state, r_0, deltar_0, n_corr=10):
	""" metropolis hastings sampling with automated step size adjustment """
	
	deltar_1 = torch.clip(deltar_0 + 0.01*rnd.normal(rng), a_min=0.005, a_max=0.5)

	acc = []
	for deltar in [deltar_0, deltar_1]:
		
		for _ in torch.arange(n_corr):
			rng, rng_alpha = rnd.split(rng, 2)

			p_0 = (torch.exp(state.apply_fn(state.params, r_0))**2)  			# ❗can make more efficient with where statement at end
			
			r_1 = r_0 + rnd.normal(rng, r_0.shape, dtype=r_0.dtype)*0.02
			
			p_1 = torch.exp(state.apply_fn(state.params, r_1))**2
			p_1 = torch.where(torch.isnan(p_1), 0., p_1)

			p_mask = (p_1/p_0) > rnd.uniform(rng_alpha, p_1.shape)			# metropolis hastings
			
			r_0 = torch.where(p_mask[..., None, None], r_1, r_0)
	
		acc += [p_mask.mean()]
	
	mask = ((0.5-acc[0])**2 - (0.5-acc[1])**2) < 0.
	deltar = mask*deltar_0 + ~mask*deltar_1
	
	return r_0, (acc[0]+acc[1])/2., deltar

### Test Suite ###

def check_antisym(c, rng, r):
	n_u, n_d, = c.data.n_u, c.data.n_d
	r = r[:, :4]
	
	@partial(jax.vmap, in_axes=(0, None, None))
	def swap_rows(r, i_0, i_1):
		return r.at[[i_0,i_1], :].set(r[[i_1,i_0], :])

	@partial(jax.pmap, axis_name='dev', in_axes=(0,0))
	def _create_train_state(rng, r):
		model = c.partial(FermiNet, with_sign=True)  
		params = model.init(rng, r)['params']
		return TrainState.create(apply_fn=model.apply, params=params, tx=c.opt.tx)
	
	state = _create_train_state(rng, r)

	@partial(jax.pmap, in_axes=(0, 0))
	def _check_antisym(state, r):
		log_psi_0, sgn_0 = state.apply_fn(state.params, r)
		r_swap_u = swap_rows(r, 0, 1)
		log_psi_u, sgn_u = state.apply_fn(state.params, r_swap_u)
		log_psi_d = torch.zeros_like(log_psi_0)
		sgn_d = torch.zeros_like(sgn_0)
		if not n_d == 0:
			r_swap_d = swap_rows(r, n_u, n_u+1)
			log_psi_d, sgn_d = state.apply_fn(state.params, r_swap_d)
		return (log_psi_0, log_psi_u, log_psi_d), (sgn_0, sgn_u, sgn_d), (r, r_swap_u, r_swap_d)

	res = _check_antisym(state, r)

	(log_psi, log_psi_u, log_psi_d), (sgn, sgn_u, sgn_d), (r, r_swap_u, r_swap_d) = res
	for ei, ej, ek in zip(r[0,0], r_swap_u[0,0], r_swap_d[0,0]):
		print(ei, ej, ek)  # Swap Correct
	for lpi, lpj, lpk in zip(log_psi[0], log_psi_u[0], log_psi_d[0]):
		print(lpi, lpj, lpk)  # Swap Correct
	for lpi, lpj, lpk in zip(sgn[0], sgn_u[0], sgn_d[0]):
		print(lpi, lpj, lpk)  # Swap Correct

