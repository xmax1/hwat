






from pathlib import Path
import jax
from jax import numpy as jnp
from PIL import Image

from jax import pmap
from flax import jax_utils
from flax.training import common_utils, train_state, dynamic_scale
import optax
import paramiko
import sys
import subprocess
import wandb
from time import sleep
from functools import partial, reduce
from itertools import product
from simple_slurm import Slurm
import random
from typing import Any, Iterable
import re
from ast import literal_eval
import jax
from jax import numpy as jnp
from jax import random as rnd
from typing import Any
import optax
from flax.training.train_state import TrainState
from flax import linen as nn, jax_utils
from pprint import pprint


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
        # mask_up = mask_up[eye_mask].reshape(-1) # for when drop diagonal enforced
        mask_down = jnp.zeros((n_electrons, n_electrons))
        mask_down[electron, :] = downs
        pairwise_down_mask.append(mask_down)

    pairwise_up_mask = jnp.stack(pairwise_up_mask, axis=0)[..., None]
    pairwise_down_mask = jnp.stack(pairwise_down_mask, axis=0)[..., None]
    return pairwise_up_mask, pairwise_down_mask

from typing import Callable

def compute_metric(d:dict):
    ...
    metrics = lax.pmean(metrics, axis_name='b')

    return metric


def compute_energy():
    
    return 


n_e = 10
x = jnp.ones((2, n_e, 3))
xu, xd = x.split(2, axis=1)

def init_walker(xu, xd):
    rng = rnd.PRNGKey(c.seed)
    rng_u, rng_d = rnd.split(rng, 2)
    return jnp.concatenate([rnd.normal(rng_u, xu.shape), rnd.normal(rng_d, xd.shape)], axis=1)


class SampleState():
    """
    @struct.dataclass
    struct.PyTreeNode means jax transformations *do not affect it* eg pmap
    fn.apply << apply vs fn() << apply_fn"""
    
    def __init__(_i, step=0, deltar=0.02, corr_len=20):
        _i.acc_target = 0.5
        _i.corr_len = corr_len//2
        _i.deltar = deltar
        _i.step = step
    
    def __call__(_i, x, state:TrainState):
        _i.rng, rng_0, rng_1, rng_move = rnd.split(_i.rng, 4)

        x, acc_0 = sample(rng_0, x, state, _i.corr_len, _i.deltar)
        deltar_1 = jnp.clip(_i.deltar + 0.001*rnd.normal(rng_move))
        x, acc_1 = sample(rng_1, x, state, _i.corr_len, deltar_1)

        mask = jnp.array((_i.acc_target-acc_0)**2 < (_i.acc_target-acc_1)**2, dtype=jnp.float32)
        not_mask = ((mask-1.)*-1.)
        _i.deltar = mask*_i.deltar + not_mask*deltar_1
        return x

def sample(rng, x, state:TrainState, corr_len, deltar):

    def move(x, rng, deltar):
        x = x + rnd.normal(rng, x.shape)*deltar
        return x

    to_prob = lambda log_psi: jnp.exp(log_psi)**2
    
    p = to_prob(state(x))
    
    acc = 0.0
    for _ in range(corr_len//2):
        rng, rng_move, rng_alpha = rnd.split(rng)
        
        x_1 = move(x, rng_move, deltar)
        p_1 = to_prob(state(x_1))

        p_mask = (p_1 / p) > rnd.uniform(rng_alpha, p_1.shape)
        p = jnp.where(p_mask, p_1, p)
        p_mask = jnp.expand_dims(p, axis=(-1, -2))
        x = jnp.where(p_mask, x_1, x)

        acc += jnp.mean(p_mask)

    return x, acc


# x = init_walker(xu, xd)