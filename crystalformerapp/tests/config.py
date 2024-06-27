import os, sys
testdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.dirname(testdir)
datadir = os.path.join(rootdir, "data")
sys.path.append(os.path.join(testdir, "../src"))

import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from jax.lib import xla_bridge
# Check if GPU is available
try:
    if xla_bridge.get_backend().platform == 'gpu':
        print("GPU is available. Using GPU.")
    else:
        raise RuntimeError("No GPU available, switching to CPU.")
except RuntimeError as e:
    print(e)
    os.environ["JAX_PLATFORMS"] = "cpu"
    jax.config.update("jax_platform_name", "cpu")
    print("Changed platform to CPU.")