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

### MODEL ### https://www.notion.so/HWAT-Docs-2bd230b570cc4814878edc00753b2525#cc424dfd1320496f85fc0504b41946cd

@partial(
	nn.vmap,
	variable_axes={"params": None, "batch_stats": None},
	split_rngs={"params": False},
	in_axes=0,
	out_axes=0,
	axis_name="b"
)
class FermiNet(nn.Module):
	n_e: int = None
	n_u: int = None
	n_d: int = None
	compute_s_emb: Callable = None
	compute_p_emb: Callable = None
	n_det: int = None
	n_fb: int = None
	n_fbv: int = None
	n_pv: int = None
	n_sv: int = None
	a: jnp.ndarray = None
	with_sign: bool = False
	
	@nn.compact # params arg hidden in apply
	def __call__(_i, r: jnp.ndarray):
		
		if len(r.shape) == 1:  # jvp hack
			r = r.reshape(_i.n_e, 3)
		
		ru, rd = r[:_i.n_u], r[_i.n_u:]

		p_mask_u, p_mask_d = create_masks(_i.n_e, _i.n_u, r.dtype)

		r_s_var = _i.compute_s_emb(r)
		r_p_var = _i.compute_p_emb(r) + jnp.eye(_i.n_e)[..., None]
		# wpr(dict(r_s_var=r_s_var, r_p_var=r_p_var))

		# r_s_res = jnp.zeros((_i.n_e, _i.n_sv), dtype=r.dtype)
		# r_p_res = jnp.zeros((_i.n_e, _i.n_e, _i.n_pv), dtype=r.dtype)
		for l in range(_i.n_fb):
			r_s_var = jnp.tanh(nn.Dense(_i.n_sv, name=f's_{l}')(r_s_var)) # + r_s_res r_s_res =
			r_p_var = jnp.tanh(nn.Dense(_i.n_pv, name=f'p_{l}')(r_p_var)) # r_p_res =+ r_p_res
			r_s_var = compute_s_perm(r_s_var, r_p_var, _i.n_u, p_mask_u, p_mask_d)

		r_w = jnp.tanh(nn.Dense(_i.n_fbv//2)(r_s_var))
		r_wu = nn.Dense(_i.n_det*_i.n_u)(r_w[:_i.n_u])
		r_wd = nn.Dense(_i.n_det*_i.n_d)(r_w[_i.n_u:])
		# wpr(dict(r_w=r_w, r_wu=r_wu, r_wd=r_wd))

		orb_u = (r_wu * jnp.exp(-nn.Dense(_i.n_u*_i.n_det)(ru-_i.a))).reshape((_i.n_det, _i.n_u, _i.n_u)) # (e, f(e)) (e, (f(e))*n_det)
		orb_d = (r_wd * jnp.exp(-nn.Dense(_i.n_d*_i.n_det)(rd-_i.a))).reshape((_i.n_det, _i.n_d, _i.n_d))
		# wpr(dict(orb_u=orb_u, orb_d=orb_d))

		log_psi, sgn = logabssumdet(orb_u, orb_d)

		if _i.with_sign:
			return log_psi, sgn
		else:
			return log_psi

def compute_emb(r, terms, a=None):  
	n_e, _ = r.shape

	z = []  
	if 'r' in terms:  
		z += [r]  
	if 'r_len' in terms:  
		z += [jnp.linalg.norm(r, axis=-1, keepdims=True)]  
	if 'ra' in terms:  
		z += [compute_rrvec(r, a).reshape(n_e, -1)]  
	if 'ra_len' in terms:  
		z += [compute_rrlen(r, a).reshape(n_e, -1)]
	if 'rr' in terms:
		z += [compute_rrvec(r, r)]
	if 'rr_len' in terms:
		z += [compute_rrlen(r, r)]
	return jnp.concatenate(z, axis=-1)

compute_rrvec = lambda ri, rj: ri[:, None, :] - rj[None, :, :]

compute_rrlen = lambda ri, rj, keepdims=True: jnp.linalg.norm(compute_rrvec(ri, rj), axis=-1, keepdims=keepdims)

def compute_s_perm(r_s_v, r_p_v, n_u, p_mask_u, p_mask_d):

	mean_s_u = r_s_v[:n_u].mean(axis=0, keepdims=True).repeat(r_s_v.shape[0], axis=0)
	mean_s_d = r_s_v[n_u:].mean(axis=0, keepdims=True).repeat(r_s_v.shape[0], axis=0)

	sum_p_u = jnp.expand_dims(p_mask_u, axis=-1)*r_p_v
	sum_p_d = jnp.expand_dims(p_mask_d, axis=-1)*r_p_v 

	return jnp.concatenate((r_s_v, mean_s_u, mean_s_d, sum_p_u.mean(0), sum_p_d.mean(0)), axis=-1)

def logabssumdet(orb_u, orb_d=None):
	
	xs = [orb_u, orb_d] if not orb_d is None else [orb_u]
	
	dets = [x.reshape(-1) for x in xs if x.shape[-1] == 1]
	dets = reduce(lambda a,b: a*b, dets) if len(dets)>0 else 1

	slogdets = [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1]
	
	if len(slogdets) > 0: # at least 2 electon in at least 1 orbital
		sign_in, logdet = reduce(lambda a,b: (a[0]*b[0], a[1]+b[1]), slogdets)
		maxlogdet = jnp.max(logdet)
		det = sign_in * dets * jnp.exp(logdet-maxlogdet)
	else:
		maxlogdet = 0
		det = dets

	psi_ish = jnp.sum(det)
	sgn_psi = jnp.sign(psi_ish)
	log_psi = jnp.log(jnp.abs(psi_ish)) + maxlogdet
	return log_psi, sgn_psi

def create_masks(n_e, n_u, dtype):
	""" up with down and down with up """
	n_d = n_e - n_u
	e_i = jnp.arange(0, n_e)[:, None]
	e_j = jnp.arange(0, n_e)[None, :]
	m_u = (e_i < n_u) * (e_j >= n_u)
	m_d = (e_i >= n_d) * (e_j < n_u)
	return m_u.astype(dtype), m_d.astype(dtype)

def maybe_put_param_in_dict(params):
	return {'params':params} if not 'params' in params else params # yep 

### ENERGY ### https://www.notion.so/HWAT-Docs-2bd230b570cc4814878edc00753b2525#a017562b66f149529204bed2ae0d4bd8

def batched_cdist_l2(x1, x2):
	x1_sq = jnp.sum(x1 ** 2, axis=-1, keepdims=True)
	x2_sq = jnp.sum(x2 ** 2, axis=-1, keepdims=True)
	cdist = jnp.sqrt(jnp.swapaxes(x1_sq, -1, -2) + x2_sq \
		- jnp.sum(2 * jnp.expand_dims(x1, axis=0) * jnp.expand_dims(x2, axis=1), axis=-1))
	return cdist

# @partial(jax.vmap, in_axes=(0,None,None))
def compute_pe(x, a, a_z):
	n_a = len(a)

	e_e_dist = batched_cdist_l2(x, x)
	pe = jnp.sum(jnp.tril(1. / e_e_dist, k=-1))    # https://www.notion.so/Potential-energy-fn-ac8267aefc2343778174958fad531cb5

	a_e_dist = batched_cdist_l2(a, x)
	pe -= jnp.sum(a_z / a_e_dist)

	if n_a > 1:
		a_a_dist = batched_cdist_l2(a, a)
		weighted_a_a = (a_z[:, None] * a_z[None, :]) / a_a_dist
		unique_a_a = jnp.tril(weighted_a_a, k=-1)
		pe += jnp.sum(unique_a_a)
	return pe

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
	def __call__(_i, x):
		n_a = len(_i.a)

		e_e_dist = batched_cdist_l2(x, x)
		pe = jnp.sum(jnp.tril(1. / e_e_dist, k=-1))

		a_e_dist = batched_cdist_l2(_i.a, x)
		pe -= jnp.sum(_i.a_z / a_e_dist)

		if n_a > 1:
			a_a_dist = batched_cdist_l2(_i.a, _i.a)
			weighted_a_a = (_i.a_z[:, None] * _i.a_z[None, :]) / a_a_dist
			unique_a_a = jnp.tril(weighted_a_a, k=-1)
			pe += jnp.sum(unique_a_a)
		return pe

def compute_ke_b(state, x):
	n_b, n_e, _ = x.shape
	
	partial_state = partial(state.apply_fn, state.params)
	model = lambda x: partial_state(x).sum()
	grad = jax.grad(model)

	x = x.reshape(n_b, -1)
	n_jvp = x.shape[-1]
	eye = jnp.eye(n_jvp, dtype=x.dtype)[None, ...].repeat(n_b, axis=0)
	
	def _body_fun(i, val):
		primal, tangent = jax.jvp(grad, (x,), (eye[..., i],))  
		return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
	
	return -0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, jnp.zeros(n_b,))

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
				lst += [center[x_i%n_center] + rnd.normal(rng_i,(n_b,1,3))*std]
			return rng_n
		
		rng_n = init_spin(rng_n, n_u, lst)
		_     = init_spin(rng_n, n_d, lst)
		device_lst += [jnp.concatenate(lst, axis=1)]

	return jnp.stack(device_lst)

def move(x, rng, deltar):
	return x + rnd.normal(rng, x.shape, dtype=x.dtype)*deltar
	 
def sample(
	rng, 
	state : TrainState, 
	x,
	deltar,
	corr_len=20, 
	acc_target=0.5,
):
	rng, rng_0, rng_1, rng_move = rnd.split(rng, 4)

	x, acc_0 = sample_subset(rng_0, x, state, deltar, corr_len//2)
	deltar_1 = jnp.clip(deltar + 0.001*rnd.normal(rng_move), a_min=0.001, a_max=0.5)
	x, acc_1 = sample_subset(rng_1, x, state, deltar_1, corr_len//2)

	mask = jnp.array((acc_target-acc_0)**2 < (acc_target-acc_1)**2, dtype=jnp.float32)
	not_mask = ((mask-1.)*-1.)
	deltar = mask*deltar + not_mask*deltar_1
	v = dict(
		deltar = deltar,
		acc = (acc_1+acc_0)/2.,
		rng = rng
	)
	return x, v

to_prob = lambda log_psi: jnp.exp(log_psi)**2

@partial(jit, static_argnames=('corr_len'))
def sample_subset(rng, x, state, deltar, corr_len):
	
	p = to_prob(state.apply_fn(state.params, x))
	
	acc = 0.0
	for _ in jnp.arange(corr_len):
		rng, rng_move, rng_alpha = rnd.split(rng, 3)
		
		x_1 = move(x, rng_move, deltar)
		p_1 = to_prob(state.apply_fn(state.params, x_1))

		p_mask = (p_1 / p) > rnd.uniform(rng_alpha, p_1.shape)
		p = jnp.where(p_mask, p_1, p)
		p_mask = jnp.expand_dims(p_mask, axis=(-1, -2))
		x = jnp.where(p_mask, x_1, x)

		acc += jnp.mean(p_mask)

	return x, acc/corr_len


### Test Suite ###

def check_antisym(c, rng, r):
	n_u, n_d, = c.data.n_u, c.data.n_d
	
	@partial(vmap, in_axes=(0, None, None))
	def swap_rows(r, i_0, i_1):
		return r.at[[i_0,i_1], :].set(r[[i_1,i_0], :])

	@partial(jax.pmap, axis_name='dev', in_axes=(0,0))
	def _create_train_state(rng, r):
		model = c.partial(FermiNet, with_sign=True)  
		params = model.init(rng, r)
		return TrainState.create(apply_fn=model.apply, params=params, tx=c.opt.tx)
	
	state = _create_train_state(rng, r)

	@partial(pmap, in_axes=(0, 0))
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

"""