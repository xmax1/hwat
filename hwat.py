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

### MODEL ### https://www.notion.so/HWAT-Docs-2bd230b570cc4814878edc00753b2525#cc424dfd1320496f85fc0504b41946cd

class FermiNet(nn.Module):
	n_e: int = None
	n_u: int = None
	n_d: int = None
	compute_s_emb: Callable = None
	compute_p_emb: Callable = None
	compute_s_perm: Callable = None
	n_det: int = None
	n_fb: int = None
	n_fb_out: int = None
	n_pv: int = None
	n_sv: int = None

	@nn.compact # params arg hidden in apply
	def __call__(_i, x):
		
		if len(x.shape) == 1:  # jvp hack
			x = x.reshape(_i.n_e, 3)
			
		p_mask_u, p_mask_d = create_masks(_i.n_e, _i.n_u)

		xu, xd = jnp.split(x, [_i.n_u,], axis=0)
		x_s_var = _i.compute_s_emb(x)
		x_p_var = _i.compute_p_emb(x) + jnp.eye(_i.n_e)[..., None]
		wpr(dict(x_s_var=x_s_var, x_p_var=x_p_var))

		x_s_res = x_p_res = 0.
		for _ in range(_i.n_fb):
			x_p_var = x_p_res = nn.tanh(nn.Dense(_i.n_pv)(x_p_var)) + x_p_res
			x_s_var = _i.compute_s_perm(x_s_var, x_p_var, p_mask_u, p_mask_d)
			x_s_var = x_s_res = nn.tanh(nn.Dense(_i.n_sv)(x_s_var)) + x_s_res
			wpr(dict(x_p_var=x_p_var, x_s_var=x_s_var))

		x_w = nn.tanh(nn.Dense(_i.n_fb_out)(x_s_var))
		x_wu, x_wd = jnp.split(x_w, [_i.n_u,], axis=0)
		x_wu = nn.tanh(nn.Dense(_i.n_det*_i.n_u)(x_wu))
		x_wd = nn.tanh(nn.Dense(_i.n_det*_i.n_d)(x_wd))
		wpr(dict(x_w=x_w, x_wu=x_wu, x_wd=x_wd))

		orb_u = jnp.stack((x_wu * jnp.exp(-nn.Dense(_i.n_u*_i.n_det)(-xu))).split(_i.n_det, axis=-1)) # (e, f(e)) (e, (f(e))*n_det)
		orb_d = jnp.stack((x_wd * jnp.exp(-nn.Dense(_i.n_d*_i.n_det)(-xd))).split(_i.n_det, axis=-1))
		wpr(dict(orb_u=orb_u, orb_d=orb_d))

		log_psi, sgn = logabssumdet(orb_u, orb_d)
		return log_psi

compute_rvec = lambda x0, x1: \
        jnp.expand_dims(x0, -2) - jnp.expand_dims(x1, -3)

compute_r = lambda x0, x1, keepdims=True: \
        jnp.linalg.norm(compute_rvec(x0, x1), axis=-1, keepdims=keepdims)

l1_norm_keep = lambda x: jnp.linalg.norm(x, axis=-1, keepdims=True)

def compute_s_perm(x, x_p, p_mask_u, p_mask_d, n_u):
    n_e, _, _ = x_p.shape
    n_d = n_e - n_u

    xu, xd = jnp.split(x, [n_u,], axis=0)
    mean_xu = jnp.repeat(jnp.mean(xu, axis=0, keepdims=True), n_e, axis=0)
    mean_xd = jnp.repeat(jnp.mean(xd, axis=0, keepdims=True), n_e, axis=0)

    x_p = jnp.expand_dims(x_p, axis=0)
    sum_p_u = (p_mask_u * x_p).sum((1, 2)) / float(n_u)
    sum_p_d = (p_mask_d * x_p).sum((1, 2)) / float(n_d)

    # wpr(dict(x=x, mean_xu=mean_xu, mean_xd=mean_xd, sum_p_u=sum_p_u, sum_p_d=sum_p_d))
    return jnp.concatenate((x, mean_xu, mean_xd, sum_p_u, sum_p_d), axis=-1)

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

def create_masks(n_electrons, n_up):
    import numpy as jnp
    ups = jnp.ones(n_electrons)
    ups[n_up:] = 0.
    downs = (ups-1.)*-1.

    pairwise_up_mask = []
    pairwise_down_mask = []
    for electron in range(n_electrons):
        mask_up = jnp.zeros((n_electrons, n_electrons))
        mask_up[electron, :] = ups
        pairwise_up_mask.append(mask_up)
        mask_down = jnp.zeros((n_electrons, n_electrons))
        mask_down[electron, :] = downs
        pairwise_down_mask.append(mask_down)

    pairwise_up_mask = jnp.stack(pairwise_up_mask, axis=0)[..., None]
    pairwise_down_mask = jnp.stack(pairwise_down_mask, axis=0)[..., None]
    return pairwise_up_mask, pairwise_down_mask

def maybe_put_param_in_dict(params):
    return {'params':params} if not 'params' in params else params # yep 

### ENERGY ### https://www.notion.so/HWAT-Docs-2bd230b570cc4814878edc00753b2525#a017562b66f149529204bed2ae0d4bd8

def batched_cdist_l2(x1, x2):
    x1_sq = jnp.sum(x1 ** 2, axis=-1, keepdims=True)
    x2_sq = jnp.sum(x2 ** 2, axis=-1, keepdims=True)
    cdist = jnp.sqrt(jnp.swapaxes(x1_sq, -1, -2) + x2_sq \
        - jnp.sum(2 * jnp.expand_dims(x1, axis=0) * jnp.expand_dims(x2, axis=1), axis=-1))
    return cdist

@partial(jax.vmap, in_axes=(0,None,None))
def compute_pe(x, a, a_z):
    n_a = len(a)

    e_e_dist = batched_cdist_l2(x, x)
    pe = jnp.sum(jnp.tril(1. / e_e_dist, k=-1))

    a_e_dist = batched_cdist_l2(a, x)
    pe -= jnp.sum(a_z / a_e_dist)

    if n_a > 1:
        a_a_dist = batched_cdist_l2(a, a)
        weighted_a_a = (a_z[:, None] * a_z[None, :]) / a_a_dist
        unique_a_a = jnp.tril(weighted_a_a, k=-1)
        pe += jnp.sum(unique_a_a)
    return pe

def create_compute_ke(model):
    """
    NB  batch input x: (n_b, n_e, 3)"""

    def _compute_ke(params, x):
        
        params = maybe_put_param_in_dict(params)
        
        n_b, n_e, _ = x.shape
        x = x.reshape(n_b, -1)
        n_jvp = x.shape[-1]

        @jax.vmap
        def _model_apply(x):
            out = model.apply(params, x)
            return out

        model_apply = lambda x: _model_apply(x).sum()

        eye = jnp.eye(n_jvp, dtype=x.dtype)[None, ...].repeat(n_b, axis=0)
        grad = jax.grad(model_apply)
        
        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad, (x,), (eye[..., i],))  
            wpr(dict(i=i, primal=primal, tanget=tangent, val=val))
            return val + (primal[:, i]**2).squeeze() + (tangent[:, i]).squeeze()
        
        return -0.5 * jax.lax.fori_loop(0, n_jvp, _body_fun, jnp.zeros(n_b,))
    return _compute_ke

### SAMPLING ### 

def init_walker(rng, n_b, n_u, n_d, center, std=0.1, ):
    lst = []
    n_center = len(center)
    
    def init_spin(rng, n, lst):
        for x_i in range(n):
            rng, rng_i = rnd.split(rng, 2)
            lst += [center[x_i%n_center] + rnd.normal(rng_i,(n_b,1,3))*std]
        return rng
    
    rng = init_spin(rng, n_u, lst)
    rng = init_spin(rng, n_d, lst)
    return jnp.concatenate(lst, axis=1)

class SampleState(): # https://www.notion.so/HWAT-Docs-2bd230b570cc4814878edc00753b2525#c5bf62a754e34116bb7696e88cb6a746
    """
    @struct.dataclass
    struct.PyTreeNode means jax transformations *do not affect it* eg pmap
    fn.apply << apply vs fn() << apply_fn"""
    
    # clean this to dataclass structure
    def __init__(_i, 
        rng, 
        step        = 0, 
        deltar      = 0.02, 
        corr_len    = 20,
        state : TrainState      = None, 
        model : nn.Module       = None,
    ):

        _i.acc_target = 0.5
        _i.corr_len = corr_len//2
        _i.deltar = deltar
        _i.step = step
        _i.rng = rng
        _i.model = state.apply_fn if model is None else model.apply
        
    def __call__(_i, params, x:jnp.ndarray):
        
        params = maybe_put_param_in_dict(params)
        model = partial(_i.model, params) # variables= is the first argument, but named setting no worky
        
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