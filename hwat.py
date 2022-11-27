






from pathlib import Path
import jax
from jax import numpy as jnp
from PIL import Image

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