import os, sys
testdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.dirname(testdir)
datadir = os.path.join(rootdir, "data")
sys.path.append(os.path.join(testdir, "../src"))

import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
