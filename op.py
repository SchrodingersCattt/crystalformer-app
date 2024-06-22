import gdown

# a file
url = "https://drive.google.com/file/d/1xnN9FMF2xRMe1lX9nWCUybuerp6ea2E9/view?usp=sharing"
output = "epoch_009800.pkl"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

import sys
sys.path.append('./src/')

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
# from hydra import initialize, compose


import checkpoint
from transformer import make_transformer
import yaml

with open("/content/CrystalFormer/model/config.yaml") as stream:
  args = yaml.safe_load(stream)

class MyObject:
    def __init__(self, d=None):
      for key, value in d.items():
          setattr(self, key, value)

args = MyObject(args)

key = jax.random.PRNGKey(42)
params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max,
                     args.h0_size,
                     4, 8,
                     32, args.model_size, args.embed_size,
                     args.atom_types, args.wyck_types,
                     0.3)

print("\n========== Load checkpoint==========")
restore_path = "./"
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path)
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    params = ckpt["params"]
else:
    print("No checkpoint file found. Start from scratch.")

print ("# of transformer params", ravel_pytree(params)[0].size)

import numpy as np
import ipywidgets as widgets
from pymatgen.core import Structure, Lattice
from weas_widget import WeasWidget

from sample import sample_crystal, make_update_lattice
from elements import element_dict, element_list
from scripts.awl2struct import get_struct_from_lawx
from utils import letter_to_number
from mcmc import make_mcmc_step
from loss import make_loss_fn

jax.config.update("jax_enable_x64", True) # to get off compilation warning, and to prevent sample nan lattice

loss_fn, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer, args.lamb_a, args.lamb_w, args.lamb_l)

# @title Interactive Sample (~30s) <a name="unconditional-chain"></a> {display-mode: "form"}
# @markdown Specify the space group, temperature, elements and so on  \\

# @markdown **seed**: random seed to sample the crystal structure \\
# @markdown **spacegroup**: control the space group of generated crystals \\
# @markdown **temperature**: modifies the probability distribution      \\
# @markdown **T1**: the temperature of sampling the first atom type    \\
# @markdown **elements**:  control the elements in the generating process. Note that you need to enter the elements separated by spaces, i.e., `Ba Ti O`, if the elements string is none, the model will not limit the elements  \\
# @markdown **wyckoff**:  control the Wyckoff letters in the generation. Note that you need to enter the Wyckoff letters separated by spaces, i.e., `a c`, if the Wyckoff is none, the model will not limit the wyckoff letter.  \\
# @markdown **nsweeps**: control the steps of mcmc to refine the generated structures




seed = 42 # @param {type:"slider", min:0, max:100, step:1}
spacegroup = 216 # @param {type:"slider", min:1, max:230, step:1}
temperature = 1  # @param {type:"slider", min:0.5, max:1.5, step:0.1}
T1 = 1000 # @param {type:"slider", min:100, max:100000000, step:100}
top_p = 1
elements = ""  # @param {type:"string"}
wyckoff = "a c"  # @param {type:"string"}
nsweeps = 5 # @param {type:"slider", min:0, max:20, step:1}

mc_width = 0.1
n_sample = 1


elements = elements.split()
if elements is not None:
    idx = [element_dict[e] for e in elements]
    atom_mask = [1] + [1 if a in idx else 0 for a in range(1, args.atom_types)]
    atom_mask = jnp.array(atom_mask)
    print ('sampling structure formed by these elements:', elements)
    print (atom_mask)
    atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
else:
    atom_mask = jnp.zeros((args.atom_types), dtype=int) # we will do nothing to a_logit in sampling
    atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)

wyckoff = wyckoff.split()
if wyckoff is not None:
  idx = [letter_to_number(w) for w in wyckoff]
  # padding 0 until the length is args.n_max
  w_mask = idx + [0]*(args.n_max -len(idx))
  print(w_mask)
  w_mask = jnp.array(w_mask, dtype=int)
  print ('sampling structure formed by these Wyckoff positions:', wyckoff)
  print (w_mask)
else:
  w_mask = None # we will do nothing to w_logit in sampling

temperature = jnp.array(temperature, dtype=float)
constraints = jnp.arange(0, args.n_max, 1)

mc_steps = nsweeps * args.n_max
print("mc_steps", mc_steps)
mcmc = make_mcmc_step(params, n_max=args.n_max, atom_types=args.atom_types, atom_mask=atom_mask, constraints=constraints)
update_lattice = make_update_lattice(transformer, params, args.atom_types, args.Kl, args.top_p, args.temperature)

key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
XYZ, A, W, M, L = sample_crystal(subkey, transformer, params, args.n_max, n_sample, args.atom_types, args.wyck_types, args.Kx, args.Kl, spacegroup, w_mask, atom_mask, top_p, temperature, T1, constraints)
x = (G, L, XYZ, A, W)
key, subkey = jax.random.split(key)
x, acc = mcmc(logp_fn, x_init=x, key=subkey, mc_steps=mc_steps, mc_width=mc_width)
print("acc", acc)

G, L, XYZ, A, W = x
key, subkey = jax.random.split(key)
L = update_lattice(subkey, G, XYZ, A, W)


XYZ = np.array(XYZ)
A = np.array(A)
W = np.array(W)
L = np.array(L)

G = np.array([spacegroup for i in range(len(L))])

structures = [get_struct_from_lawx(g, l, a, w, xyz) for g, l, a, w, xyz in zip(G, L, A, W, XYZ)]
structures = [Structure.from_dict(_) for _ in structures]


viewer1 = WeasWidget()
viewer1.from_pymatgen(structures[0])
# avr is atoms_viewer
viewer1.avr.boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]
viewer1.avr.model_style = 1
viewer1.avr.show_bonded_atoms = True
viewer1.avr.color_type = "VESTA"
viewer1
