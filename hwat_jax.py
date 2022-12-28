import jax
from jax import numpy as jnp
from functools import reduce, partial
from jax import random as rnd
from typing import Any
import optax
from flax.training.train_state import TrainState
from typing import Callable
from flax import linen as nn
from jax import vmap, jit, pmap
import functools

### model ###

@partial(
	nn.vmap, 
	in_axes=0, 
	out_axes=0, 
	variable_axes={'params': None}, 
	split_rngs={'params': False}
) 																							# https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.vmap.html
class FermiNet(nn.Module):
	n_e: int = None
	n_u: int = None
	n_d: int = None
	n_det: int = None
	n_fb: int = None
	n_fbv: int = None
	n_pv: int = None
	n_sv: int = None
	a: jnp.ndarray = None
	with_sign: bool = False
	
	@nn.compact # params arg hidden in apply
	def __call__(_i, r: jnp.ndarray):
		eye = jnp.eye(_i.n_e)[..., None]
		
		if len(r.shape) == 1:  # jvp hack
			r = r.reshape(_i.n_e, 3)
		
		ra = r[:, None, :] - _i.a[None, :, :] # (r_i, a_j, 3)
		ra_len = jnp.linalg.norm(ra, axis=-1, keepdims=True) # (r_i, a_j, 1)
		
		rr = (r[None, :, :] - r[:, None, :])
		rr_len = jnp.linalg.norm(rr+eye,axis=-1,keepdims=True) * (jnp.ones((_i.n_e,_i.n_e,1))-eye)

		s_v = jnp.concatenate([ra, ra_len], axis=-1).reshape(_i.n_e, -1)
		p_v = jnp.concatenate([rr, rr_len], axis=-1)
		
		# print(s_v.mean(), p_v.mean())

		for l in range(_i.n_fb):
			sfb_v = [jnp.tile(_v.mean(axis=0)[None, :], (_i.n_e, 1)) for _v in s_v.split([_i.n_u,], axis=0)]
			pfb_v = [_v.mean(axis=0) for _v in p_v.split([_i.n_u,], axis=0)]
			
			s_v = jnp.concatenate(sfb_v+pfb_v+[s_v,], axis=-1) 
   
			# print(l, s_v.mean())
			s_v = nn.tanh(nn.Dense(_i.n_sv, bias_init=jax.nn.initializers.uniform(0.01))(s_v)) + (s_v if (s_v.shape[-1]==_i.n_sv) else 0.)

			if not (l == (_i.n_fb-1)):
				p_v = nn.tanh(nn.Dense(_i.n_pv, bias_init=jax.nn.initializers.uniform(0.01))(p_v)) + (p_v if (p_v.shape[-1]==_i.n_pv) else 0.)
	
		s_u, s_d = s_v.split([_i.n_u,], axis=0)

		s_u = nn.Dense(_i.n_sv//2, bias_init=jax.nn.initializers.uniform(0.01))(s_u)
		s_d = nn.Dense(_i.n_sv//2, bias_init=jax.nn.initializers.uniform(0.01))(s_d)
		
		s_wu = nn.Dense(_i.n_u, bias_init=jax.nn.initializers.uniform(0.01))(s_u)
		s_wd = nn.Dense(_i.n_d, bias_init=jax.nn.initializers.uniform(0.01))(s_d)
		
		
		assert s_wd.shape == (_i.n_d, _i.n_d)

		ra_u, ra_d = ra.split([_i.n_u,], axis=0)

		# Single parameter on norm
		# exp_u = jnp.tile(jnp.linalg.norm(ra_u, axis=-1)[..., None], (1, 1, 3))
		# exp_d = jnp.tile(jnp.linalg.norm(ra_d, axis=-1)[..., None], (1, 1, 3))
		# exp_u = nn.Dense(_i.n_u, use_bias=False)(exp_u)
		# exp_d = nn.Dense(_i.n_d, use_bias=False)(exp_d)

		exp_u = jnp.linalg.norm(ra_u, axis=-1)[..., None]
		exp_d = jnp.linalg.norm(ra_d, axis=-1)[..., None]
		# print('exp', exp_u.mean(), exp_d.mean())
		assert exp_d.shape == (_i.n_d, _i.a.shape[0], 1)

		# print(exp_d.shape)
		orb_u = (s_wu * (jnp.exp(-exp_u).sum(axis=1)))[None, ...]
		orb_d = (s_wd * (jnp.exp(-exp_d).sum(axis=1)))[None, ...]

		
		# print('exp', orb_u.mean(), orb_d.mean())
		assert orb_u.shape == (1, _i.n_u, _i.n_u)

		log_psi, sgn = logabssumdet([orb_u, orb_d])

		if _i.with_sign:
			return log_psi, sgn
		else:
			return log_psi.squeeze()


compute_vv = lambda v_i, v_j: jnp.expand_dims(v_i, axis=-2)-jnp.expand_dims(v_j, axis=-3)


def compute_emb(r, terms, a=None):  
	n_e, _ = r.shape
	eye = jnp.eye(n_e)[..., None]

	z = []  
	if 'r' in terms:  
		z += [r]  
	if 'r_len' in terms:  
		z += [jnp.linalg.norm(r, axis=-1, keepdims=True)]  
	if 'ra' in terms:  
		z += [(r[:, None, :] - a[None, ...]).reshape(n_e, -1)]  
	if 'ra_len' in terms:  
		z += [jnp.linalg.norm(r[:, None, :] - a[None, ...], axis=-1)]
	if 'rr' in terms:
		z += [compute_vv(r, r)]
	if 'rr_len' in terms:  # 2nd order derivative of norm is undefined, so use eye
		z += [jnp.linalg.norm(compute_vv(r, r)+eye, axis=-1, keepdims=True) * (jnp.ones((n_e,n_e,1))-eye)]
	return jnp.concatenate(z, axis=-1)

def logabssumdet(xs):
	
	dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]						# in case n_u or n_d=1, no need to compute determinant
	dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1.					# take product of these cases
	maxlogdet = 0.																# initialised for sumlogexp trick (for stability)
	det = dets																	# if both cases satisfy n_u or n_d=1, this is the determinant
	
	slogdets = [jnp.linalg.slogdet(x) for x in xs if x.shape[-1]>1] 			# otherwise take slogdet
	if len(slogdets)>0: 
		sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)  # take product of n_u or n_d!=1 cases
		maxlogdet = jnp.max(logdet)												# adjusted for new inputs
		det = sign_in * dets * jnp.exp(logdet-maxlogdet)						# product of all these things is determinant
	
	psi_ish = jnp.sum(det)
	sgn_psi = jnp.sign(psi_ish)
	log_psi = jnp.log(jnp.abs(psi_ish)) + maxlogdet
	return log_psi, sgn_psi

### energy ###

def compute_pe_b(r, a=None, a_z=None):

	rr = jnp.expand_dims(r, -2) - jnp.expand_dims(r, -3)
	rr_len = jnp.linalg.norm(rr, axis=-1)
	pe_rr = jnp.tril(1./rr_len, k=-1).sum((1,2))

	if not (a is None):
		a, a_z = a[None, :, :], a_z[None, None, :]
		ra = jnp.expand_dims(r, -2) - jnp.expand_dims(a, -3)
		ra_len = jnp.linalg.norm(ra, axis=-1)
		pe_ra = (a_z/ra_len).sum((1,2))   
	
		if len(a) > 1:  # len(a) = n_a
			raise NotImplementedError
	return (pe_rr - pe_ra).squeeze()


def compute_ke_b(state, r):

	grad_fn = jax.grad(lambda r: state.apply_fn(state.params, r).sum())

	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim
	r = r.reshape(n_b, n_jvp)
	eye = jnp.eye(n_jvp, dtype=r.dtype)[None, ...].repeat(n_b, axis=0)

	def _body_fun(i, val):
		primal, tangent = jax.jvp(grad_fn, (r,), (eye[..., i],))  
		return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()

	return (- 0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, jnp.zeros(n_b,))).squeeze()

### sampling ###

def keep_around_points(r, points, l=1.):
	""" points = center of box each particle kept inside. """
	""" l = length side of box """
	r = r - points[None, ...]
	r = r/l
	r = jnp.fmod(r, 1.)
	r = r*l
	r = r + points[None, ...]
	return r

def get_center_points(n_e, center, _r_cen=None):
	""" from a set of centers, selects in order where each electron will start """
	""" loop concatenate pattern """
	for r_i in range(n_e):
		_r_cen = center[[r_i % len(center)]] if _r_cen is None else jnp.concatenate([_r_cen, center[[r_i % len(center)]]])
	return _r_cen

def init_r(rng, n_b, n_e, center_points, std=0.1):
	""" init r on different gpus with different rngs """
	""" loop concatenate pattern """
	sub_r = [center_points + rnd.normal(rng_i,(n_b,n_e,3))*std for rng_i in rng]
	return jnp.stack(sub_r) if len(sub_r)>1 else sub_r[0][None, ...]

def sample_b(rng, state, r_0, deltar_0, n_corr=10):
	""" metropolis hastings sampling with automated step size adjustment """
	
	deltar_1 = jnp.clip(deltar_0 + 0.01*rnd.normal(rng), a_min=0.005, a_max=0.5)

	acc = []
	for deltar in [deltar_0, deltar_1]:
		
		for _ in jnp.arange(n_corr):
			rng, rng_alpha = rnd.split(rng, 2)

			p_0 = (jnp.exp(state.apply_fn(state.params, r_0))**2)  			# ❗can make more efficient with where statement at end
			
			r_1 = r_0 + rnd.normal(rng, r_0.shape, dtype=r_0.dtype)*0.02
			
			p_1 = jnp.exp(state.apply_fn(state.params, r_1))**2
			p_1 = jnp.where(jnp.isnan(p_1), 0., p_1)

			p_mask = (p_1/p_0) > rnd.uniform(rng_alpha, p_1.shape)			# metropolis hastings
			
			r_0 = jnp.where(p_mask[..., None, None], r_1, r_0)
	
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
		log_psi_d = jnp.zeros_like(log_psi_0)
		sgn_d = jnp.zeros_like(sgn_0)
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

