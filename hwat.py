import jax
from jax import numpy as jnp
from functools import reduce, partial
from jax import random as rnd
from typing import Any
import optax
from flax.training.train_state import TrainState
from utils import wpr
from typing import Callable
from flax import linen as nn
from jax import vmap, jit, pmap
import functools

### MODEL ### https://www.notion.so/HWAT-Docs-2bd230b570cc4814878edc00753b2525#cc424dfd1320496f85fc0504b41946cd


# b_init = lambda _, shape: jnp.zeros(shape, jnp.complex64)

@partial(
	nn.vmap, 
	in_axes=0, 
	out_axes=0, 
	variable_axes={'params': None}, 
	split_rngs={'params': False}
) # https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.vmap.html
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
		
		for l in range(_i.n_fb):
			sfb_v = [jnp.tile(_v.mean(axis=0)[None, :], (_i.n_e, 1)) for _v in s_v.split([_i.n_u,], axis=0)]
			pfb_v = [_v.mean(axis=0) for _v in p_v.split([_i.n_u,], axis=0)]
			
			s_v = jnp.concatenate(sfb_v+pfb_v+[s_v,], axis=-1)   
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

		assert exp_d.shape == (_i.n_d, _i.a.shape[0], 1)

		# print(exp_d.shape)
		orb_u = (s_wu * (jnp.exp(-exp_u).sum(axis=1)))[None, ...]
		orb_d = (s_wd * (jnp.exp(-exp_d).sum(axis=1)))[None, ...]

		assert orb_u.shape == (1, _i.n_u, _i.n_u)

		log_psi, sgn = logdet_matmul([orb_u, orb_d])

		if _i.with_sign:
			return log_psi, sgn
		else:
			return log_psi.squeeze()



def logdet_matmul(xs):
	slogdets = [jnp.linalg.slogdet(x) for x in xs]
	
	sign_in, slogdet = functools.reduce(lambda a, b: (a[0] * b[0], a[1] + b[1]), slogdets)
	max_idx = jnp.argmax(slogdet)
	slogdet_max = slogdet[max_idx]
	det = sign_in * jnp.exp(slogdet-slogdet_max)
	
	result = jnp.sum(det)
	sign_out = jnp.sign(result)
	slog_out = jnp.log(jnp.abs(result)) + slogdet_max
	return slog_out, sign_out


def logabssumdet(orb_u, orb_d=None):
	
	xs = [orb_u] if orb_d is None else [orb_u, orb_d] 
	
	dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]
	dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1.

	slogdets = [jnp.linalg.slogdet(x) for x in xs if x.shape[-1]>1]
	
	print(slogdets[0][0].shape)
	if len(slogdets)>0: # at least 2 electon in at least 1 orbital
		sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)
		maxlogdet = jnp.max(logdet)
		det = sign_in * dets * jnp.exp(logdet-maxlogdet)
	else:
		maxlogdet = 0.
		det = dets

	psi_ish = jnp.sum(det)
	sgn_psi = jnp.sign(psi_ish)
	log_psi = jnp.log(jnp.abs(psi_ish)) + maxlogdet
	return log_psi, sgn_psi


def compute_pe(r, a=None, a_z=None):
	n_a = len(a)
	
	rr = jnp.expand_dims(r, -2) - jnp.expand_dims(r, -3)
	rr_len = jnp.linalg.norm(rr, axis=-1)
	pe_rr = jnp.tril(1./rr_len, k=-1).sum((1,2))

	if not (a is None):
		a, a_z = a[None, :, :], a_z[None, None, :]
		ra = jnp.expand_dims(r, -2) - jnp.expand_dims(a, -3)
		ra_len = jnp.linalg.norm(ra, axis=-1)
		pe_ra = (a_z/ra_len).sum((1,2))   
	
		if n_a > 1:
			raise NotImplementedError
	return (pe_rr - pe_ra).squeeze()


def compute_ke_b(state, r):
	n_b, n_e, _ = r.shape
	
	model = lambda r: state.apply_fn(state.params, r).sum()
	grad = jax.grad(model)

	r = r.reshape(n_b, -1)
	n_jvp = r.shape[-1]
	eye = jnp.eye(n_jvp, dtype=r.dtype)[None, ...].repeat(n_b, axis=0)
	
	def _body_fun(i, val):
		primal, tangent = jax.jvp(grad, (r,), (eye[..., i],))  
		return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	
	return (- 0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, jnp.zeros(n_b,))).squeeze()

### SAMPLING ### 

def init_walker(rng, n_b, n_u, n_d, center, std=0.1):
	
	n_center = len(center)
	rng = [rng] if len(rng.shape) == 1 else rng
	
	device_lst = []
	for rng_n in rng:
		lst = []
		
		def init_spin(rng_n, n, lst):
			for x_i in range(n):
				rng_n, rng_i = rnd.split(rng_n, 2)
				lst += [center[x_i%n_center] + rnd.normal(rng_i,(n_b,1,3))]
			return rng_n
		
		rng_n = init_spin(rng_n, n_u, lst)
		_     = init_spin(rng_n, n_d, lst)
		device_lst += [jnp.concatenate(lst, axis=1)]

	return jnp.stack(device_lst)

def move(r, rng, deltar):
	return r + rnd.normal(rng, r.shape, dtype=r.dtype)*deltar
	 



### Test Suite ###

def check_antisym(c, rng, r):
	n_u, n_d, = c.data.n_u, c.data.n_d
	r = r[:, :4]
	
	@partial(vmap, in_axes=(0, None, None))
	def swap_rows(r, i_0, i_1):
		return r.at[[i_0,i_1], :].set(r[[i_1,i_0], :])

	@partial(jax.pmap, axis_name='dev', in_axes=(0,0))
	def _create_train_state(rng, r):
		model = c.partial(FermiNet, with_sign=True)  
		params = model.init(rng, r)['params']
		return TrainState.create(apply_fn=model.apply, params=params, tx=c.opt.tx)
	
	state = _create_train_state(rng, r)

	@partial(pmap, in_axes=(None, 0))
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





"""



def sample_i(
	rng, 
	state:TrainState, 
	r,
	deltar,
):
	rng, rng_0, rng_1, rng_move = rnd.split(rng, 4)

	_r, acc_0 = sample_subset(rng_0, state, r, deltar, 20)
	deltar_1 = jnp.clip(deltar + 0.01*rnd.normal(rng_move), a_min=0.005, a_max=0.5)
	_r, acc_1 = sample_subset(rng_1, state, _r, deltar_1, 20)

	mask = (0.5-acc_0)**2 < (0.5-acc_1)**2
	deltar = mask*deltar +  jnp.logical_not(mask)*deltar_1
	v = dict(
		deltar = deltar,
		acc = (acc_1+acc_0)/2.,
		rng = rng, 
		mask = mask,
		r = _r,
		r_og = r,
	)
	return _r, (acc_1+acc_0)/2., deltar



to_prob = lambda log_psi: jnp.exp(log_psi)**2

def sample_subset(rng, state, r_0, deltar, corr_len):
	p_0 = to_prob(state.apply_fn(state.params, r_0))
	
	acc = 0.0
	for i in jnp.arange(1, corr_len+1):
		rng, rng_move, rng_alpha = rnd.split(rng, 3)
		
		r_1 = move(r_0, rng_move, deltar)
		p_1 = to_prob(state.apply_fn(state.params, r_1))

		p_mask = (p_1 / p_0) > rnd.uniform(rng_alpha, p_1.shape)
		p_0 = jnp.where(p_mask, p_1, p_0)
		r_0 = jnp.where(p_mask[..., None, None], r_1, r_0)

		acc += jnp.mean(p_mask)
	return r_0, acc/i   

	
class SampleState(): # https://www.notion.so/HWAT-Docs-2bd230b570cc4814878edc00753b2525#c5bf62a754e34116bb7696e88cb6a746
	@struct.dataclass
	struct.PyTreeNode means jax transformations *do not affect it* eg pmap
	fn.apply << apply vs fn() << apply_fn
	
	# clean this to dataclass structure
	# 99% sure to be distributed MUST be defined in a pmap
	def __init__(_i, 
		rng, 
		step        = 0, 
		deltar      = 0.02, 
		corr_len    = 20
	):

		_i.acc_target = 0.5
		_i.corr_len = corr_len//2
		_i.deltar = deltar
		_i.step = step
		_i.rng = rng
		
	def __call__(_i, state:TrainState, x):
		
		model = partial(state.apply_fn, state.params)
		
		_i.rng, rng_0, rng_1, rng_move = rnd.split(_i.rng, 4)

		x, acc_0 = sample(rng_0, x, model, _i.corr_len, _i.deltar)
		deltar_1 = jnp.clip(_i.deltar + 0.001*rnd.normal(rng_move), a_min=0.001, a_max=0.5)
		x, acc_1 = sample(rng_1, x, model, _i.corr_len, deltar_1)

		mask = jnp.array((_i.acc_target-acc_0)**2 < (_i.acc_target-acc_1)**2, dtype=jnp.float32)
		not_mask = ((mask-1.)*-1.)
		_i.deltar = mask*_i.deltar + not_mask*deltar_1
		return x

def sample(rng, x, model:partial, corr_len, deltar):

	def move(x, rng, deltar):
		x = x + rnd.normal(rng, x.shape)*deltar
		return x

	to_prob = lambda log_psi: jnp.exp(log_psi)**2
	
	p = to_prob(model(x))
	
	acc = 0.0
	for _ in range(corr_len//2):
		rng, rng_move, rng_alpha = rnd.split(rng, 3)
		
		x_1 = move(x, rng_move, deltar)
		p_1 = to_prob(model(x_1))

		p_mask = (p_1 / p) > rnd.uniform(rng_alpha, p_1.shape)
		p = jnp.where(p_mask, p_1, p)
		p_mask = jnp.expand_dims(p, axis=(-1, -2))
		x = jnp.where(p_mask, x_1, x)

		acc += jnp.mean(p_mask)

	return x, acc


@partial(
	nn.vmap,
	in_axes=0,
	out_axes=0,
	axis_name="b",
)
class PotentialEnergy(nn.Module):
	a: jnp.ndarray
	a_z: jnp.ndarray

	@nn.compact
	def __call__(_i, r):

		n_a = len(_i.a)

		e_e_dist = batched_cdist_l2(r, r)
		pe = jnp.sum(jnp.tril(1. / e_e_dist, k=-1))    # https://www.notion.so/Potential-energy-fn-ac8267aefc2343778174958fad531cb5

		a_e_dist = batched_cdist_l2(_i.a, r)
		pe -= jnp.sum(_i.a_z / a_e_dist)

		if n_a > 1:
			print('here')
			a_a_dist = batched_cdist_l2(_i.a, _i.a)
			weighted_a_a = (_i.a_z[:, None] * _i.a_z[None, :]) / a_a_dist
			unique_a_a = jnp.tril(weighted_a_a, k=-1)
			pe += jnp.sum(unique_a_a)
		
		# rr_len = compute_rrlen(r, r).squeeze()
		# pe = jnp.tril(1./rr_len, k=-1).sum()

		# ra_len = jnp.linalg.norm(r[:, None, :] - _i.a[:, ...], axis=-1)
		# # ra_len = compute_rrlen(r,_i.a).squeeze()
		# pe     -= (_i.a_z/ra_len).sum()

		# if len(_i.a) > 1:
		# 	aa_len = compute_rrlen(_i.a, _i.a).squeeze()
		# 	weighted_a_a = (_i.a_z[:, None]*_i.a_z[None, :])/aa_len
		# 	pe += jnp.tril(weighted_a_a, k=-1).sum()
		return pe




def compute_ke(state, x):
	model = partial(state.apply_fn, state.params)

	n_b, n_e, _ = x.shape
	x = x.reshape(n_b, -1)
	n_jvp = x.shape[-1]

	@jax.vmap
	def _model_apply(x):
		out = model(x)
		return out

	model_apply = lambda x: _model_apply(x).sum()

	eye = jnp.eye(n_jvp, dtype=x.dtype)[None, ...].repeat(n_b, axis=0)
	grad = jax.grad(model_apply)
	
	def _body_fun(i, val):
		primal, tangent = jax.jvp(grad, (x,), (eye[..., i],))  
		wpr(dict(i=i, primal=primal, tanget=tangent, val=val))
		return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	
	return -0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, jnp.zeros(n_b,))



def create_masks(n_e, n_u, dtype):
	 up with down and down with up 
	um along 0th axis in fermiblock 
	eye_mask = ((jnp.eye(n_e)-1.)*-1)
	n_d = n_e - n_u
	e_i = jnp.ones((n_e, n_e))  # (i, j) \sum_j
	e_j = jnp.arange(0, n_e)[None, :]
	m_u = e_i * (e_j < n_u)  * eye_mask  # *)
	m_d = e_i * (e_j >= n_u) * eye_mask  # *(e_j < n_u))
	return m_u.astype(dtype), m_d.astype(dtype)

def compute_emb(r, terms, a=None):  
	n_e, _ = r.shape

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
		z += [compute_rrvec(r, r)]
	if 'rr_len' in terms:
		z += [compute_rrlen(r)]
	return jnp.concatenate(z, axis=-1)


def compute_rrlen(ri, rj=None):
	n_dim = ri.shape[0]
	rj_1 = ri if rj is None else rj
	rr_vec = compute_rrvec(ri, rj_1)
	if rj is None: 
		rr_vec = rr_vec * jnp.eye(n_dim)[..., None]
	return jnp.linalg.norm(rr_vec, axis=-1, keepdims=True) * ((jnp.eye(n_dim)-1.)*-1)[..., None]


compute_rrvec = lambda ri, rj: jnp.expand_dims(ri, axis=-2) - jnp.expand_dims(rj, axis=-3)


def compute_s_perm(r_s_v, r_p_v, n_u, p_mask_u, p_mask_d):

	mean_s_u = r_s_v[:n_u].mean(axis=0, keepdims=True).repeat(r_s_v.shape[0], axis=0)
	mean_s_d = r_s_v[n_u:].mean(axis=0, keepdims=True).repeat(r_s_v.shape[0], axis=0)

	sum_p_u = (p_mask_u[..., None]*r_p_v).mean(0)
	sum_p_d = (p_mask_d[..., None]*r_p_v).mean(0)

	return jnp.concatenate((r_s_v, mean_s_u, mean_s_d, sum_p_u, sum_p_d), axis=-1)



def maybe_put_param_in_dict(params):
	return {'params':params} if not 'params' in params else params # yep 






"""