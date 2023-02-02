"""
Utilities for working with JAX.
Some of these functions are taken from 
https://github.com/deepmind/ferminet/tree/jax/ferminet
"""
import functools

import jax
from jax import core

broadcast = jax.pmap(lambda x: x)

p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def replicate(pytree):
    n = jax.local_device_count()
    stacked_pytree = jax.tree_map(lambda x: jax.lax.broadcast(x, (n,)), pytree)
    return broadcast(stacked_pytree)


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(jax.lax.pmean, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(jax.lax.psum, axis_name=PMAP_AXIS_NAME)
pmax = functools.partial(jax.lax.pmax, axis_name=PMAP_AXIS_NAME)


def wrap_if_pmap(p_func):
    def p_func_if_pmap(obj):
        try:
            core.axis_frame(PMAP_AXIS_NAME)
            return p_func(obj)
        except NameError:
            return obj
    return p_func_if_pmap


pmean_if_pmap = wrap_if_pmap(pmean)
psum_if_pmap = wrap_if_pmap(psum)
pmax_if_pmap = wrap_if_pmap(pmax)



### Jax things
def import_jax():

	import jax
	from flax.core.frozen_dict import FrozenDict
	from jax import random as rnd

	def gen_rng(rng, n_device):
		""" rng generator for multi-gpu experiments """
		rng, rng_p = torch.split(rnd.split(rng, n_device+1), [1,])
		return rng.squeeze(), rng_p	

	def compute_metrix(d:dict, mode='tr', fancy=None, ignore = [], _d = {}):
		
		for k,v in d.items():
			if any([ig in k for ig in ignore+['step']]):
				continue 
			
			if not fancy is None:
				k = fancy.get(k, k)

			v = jax.device_get(v)
			
			if isinstance(v, FrozenDict):
				v = v.unfreeze()
			
			v_mean = jax.tree_map(lambda x: x.mean(), v) if not np.isscalar(v) else v
			v_std = jax.tree_map(lambda x: x.std(), v) if not np.isscalar(v) else 0.

			group = mode
			if 'grad' in k:
				group = mode + '/grad'
			elif 'param' in k:
				group += '/param'
				
			_d = collect_stats(k, v_mean, _d, p=group, suf=r'_\mu$')
			_d = collect_stats(k, v_std, _d, p=group+'/std', suf=r'_\sigma$')

		return _d

	### test things
	def test_print_fp16_no_cast():
		x = torch.ones([1], dtype='float16')
		print(x)  # FAILS

	def test_print_fp16():
		x = torch.ones([1], dtype='float16')
		x = x.astype('float16')
		print(x)  # OK

	def test_print_fp32():
		x = torch.ones([1], dtype='float16')
		x = x.astype('float16')
		x = x.astype('float32')
		print(x)  # OK

	def test_print_fp32_to_fp16_cast():
		x = torch.ones([1], dtype='float32')
		x = x.astype('float16')
		print(x)  # FAILS
  
