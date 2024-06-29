import gdown
import sys
import yaml
import os
import datetime
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

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

# from hydra import initialize, compose
import numpy as np
import ipywidgets as widgets
from pymatgen.core import Structure, Lattice
from weas_widget import WeasWidget

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crystalformerapp import src
from crystalformerapp.src.sample import sample_crystal, make_update_lattice
from crystalformerapp.src.elements import element_dict, element_list
from crystalformerapp.scripts.awl2struct import get_struct_from_lawx
from crystalformerapp.src.utils import letter_to_number
from crystalformerapp.src.mcmc import make_mcmc_step
from crystalformerapp.src.loss import make_loss_fn

from crystalformerapp.src.checkpoint import find_ckpt_filename, load_data
from crystalformerapp.src.transformer import make_transformer

output = "epoch_009800.pkl"


current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "model", "config.yaml")

with open(file_path) as stream:
    args = yaml.safe_load(stream)


class MyObject:
    def __init__(self, d=None):
      for key, value in d.items():
          setattr(self, key, value)

args = MyObject(args)
key = jax.random.PRNGKey(42)

params, transformer = make_transformer(
        key, 
        args.Nf, 
        args.Kx, 
        args.Kl, 
        args.n_max,
        args.h0_size,
        4, 
        8,
        32, 
        args.model_size, 
        args.embed_size,
        args.atom_types, 
        args.wyck_types,
        0.3
)
'''                     


params, transformer = make_transformer(
        key=jax.random.PRNGKey(42),
        Nf=5,
        Kx=16,
        Kl=4,
        n_max=21,
        h0_size=256,
        num_layers=16,
        num_heads=16,
        key_size=64,
        model_size=64,
        embed_size=32,
        atom_types=119,
        wyck_types=28,
        dropout_rate=0.5,
        widening_factor=4,
        sigmamin=1e-3
)
'''

def get_model(restore_path, params, transformer):
    pass
    
print("\n========== Load checkpoint==========")
restore_path = "/personal/crystalformerapp/crystalformerapp/model/epoch_009800.pkl"
ckpt_filename, epoch_finished = find_ckpt_filename(restore_path)
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = load_data(ckpt_filename)
    params = ckpt["params"]
else:
    print("No checkpoint file found. Start from scratch.")

print ("# of transformer params", ravel_pytree(params)[0].size)

jax.config.update("jax_enable_x64", True) # to get off compilation warning, and to prevent sample nan lattice
loss_fn, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer, args.lamb_a, args.lamb_w, args.lamb_l)


def run_op(
    spacegroup, elements, wyckoff, 
    temperature, seed, T1, nsweeps,
    nsample, tempdir='./'
):
    top_p = 1
    mc_width = 0.1
    n_sample = nsample  # batchsize equivalent

    elements = elements.split()
    if elements is not None:
        idx = [element_dict[e] for e in elements]
        atom_mask = [1] + [1 if a in idx else 0 for a in range(1, args.atom_types)]
        atom_mask = jnp.array(atom_mask)
        atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
    else:
        atom_mask = jnp.zeros((args.atom_types), dtype=int)
        atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)

    wyckoff = wyckoff.split()
    if wyckoff is not None:
        idx = [letter_to_number(w) for w in wyckoff]
        w_mask = idx + [0]*(args.n_max -len(idx))
        w_mask = jnp.array(w_mask, dtype=int)
    else:
        w_mask = None 

    temperature = jnp.array(temperature, dtype=float)
    constraints = jnp.arange(0, args.n_max, 1)

    mc_steps = nsweeps * args.n_max
    mcmc = make_mcmc_step(params, n_max=args.n_max, atom_types=args.atom_types, atom_mask=atom_mask, constraints=constraints)
    update_lattice = make_update_lattice(transformer, params, args.atom_types, args.Kl, args.top_p, args.temperature)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    output_files = []
    
    # Batch sampling
    XYZ, A, W, M, L = sample_crystal(
        subkey, 
        transformer, 
        params, 
        args.n_max, 
        n_sample, 
        args.atom_types, 
        args.wyck_types, 
        args.Kx, 
        args.Kl, 
        spacegroup, 
        w_mask, 
        atom_mask, 
        top_p, 
        temperature, 
        T1, 
        constraints)
    
    G = jnp.array([spacegroup for _ in range(len(L))])
    x = (G, L, XYZ, A, W)
    key, subkey = jax.random.split(key)
    x, acc = mcmc(logp_fn, x_init=x, key=subkey, mc_steps=mc_steps, mc_width=mc_width)

    G, L, XYZ, A, W = x
    key, subkey = jax.random.split(key)
    L = update_lattice(subkey, G, XYZ, A, W)

    XYZ = np.array(XYZ)
    A = np.array(A)
    W = np.array(W)
    L = np.array(L)
    G = np.array(G)

    # Convert batch outputs to structures
    structures = [get_struct_from_lawx(g, l, a, w, xyz) for g, l, a, w, xyz in zip(G, L, A, W, XYZ)]
    structures = [Structure.from_dict(_) for _ in structures]
    
    # Save structures to files
    for ii, structure in enumerate(structures):
        output_file = os.path.join(tempdir, f"pred_struct_{ii}.cif")
        structure.to(output_file)        
        output_files.append(output_file)

    return output_files


def run_op_gpu(
    spacegroup, elements, wyckoff, 
    temperature, seed, T1, nsweeps,
    nsample, access_key, project_id, 
    machine_type, image, tempdir 
):
    from bohrium_open_sdk import OpenSDK
    import time

    client = OpenSDK(access_key=access_key)

    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    
    cmd = (
        f"python -c \""
        f"import os; "
        f"import crystalformerapp; "
        f"from crystalformerapp import op; "
        f"op.run_op({spacegroup}, '{elements}', '{wyckoff}', "
        f"{temperature}, {seed}, {T1}, {nsweeps}, {nsample}, f'./')"
        f"\""
    )

    project_id_native = int(project_id)

    resp = client.job.submit(
        project_id=project_id_native,
        machine_type=machine_type,
        job_name="crystalformer",
        cmd=cmd,
        image_address=image,
        out_files=["*cif"],
        dataset_path=[],
        job_group_id=0,
    )
    print(resp)
    print("Job submitted. Waiting for completion...")

    job_id = resp["data"]['jobId']
    while True:
        job_info = client.job.detail(job_id)
        job_status = job_info["data"]["status"]
        if job_status == 2:
            print("Job completed!")
            client.job.download(job_id, f'{tempdir}/out.zip')
            os.system(f"unzip {tempdir}/out.zip -d {tempdir}")
            return os.path.join(tempdir, "pred_struct.cif")
        
        elif job_status == -1:
            error_message = job_info["data"].get("errorinfo", "Job failed without a specific error message.")
            print("Job failed.")
            return {"error": error_message}
        
        else:
            print("Job not done yet. Checking again after 30 seconds...")
            time.sleep(30)

