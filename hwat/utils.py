import numpy as np
from jax import numpy as jnp
import jax

cpu = jax.devices('cpus')[0]

def npify(d: dict):
    return {k:np.array(v) for k,v in d}

def move_to_host(x: jnp.ndarray, host):
    return jax.device_put(x, device=host)

    # return {k:np.array(jax.device_put(v), device=cpu) for k,v in d}
