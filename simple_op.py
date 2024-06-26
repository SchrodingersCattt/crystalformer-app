import sys
sys.path.append('./src/')
import jax
import jax.numpy as jnp
from hydra import initialize, compose
import checkpoint
from transformer import make_transformer
import numpy as np
from pymatgen.core import Structure
from time import time
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from sample import sample_crystal
from elements import element_dict
from scripts.awl2struct import get_struct_from_lawx

import os
import gradio as gr
import io
import tempfile
import shutil

def configure_jax():
    jax.config.update("jax_enable_x64", True)

def initialize_parameters(args, seed=42):
    key = jax.random.PRNGKey(seed)
    params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max,
                                           args.h0_size,
                                           4, 8,
                                           32, args.model_size, args.embed_size,
                                           args.atom_types, args.wyck_types,
                                           0.3)
    return params, transformer

def load_checkpoint(restore_path):
    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path)
    if ckpt_filename is not None:
        print(f"Load checkpoint file: {ckpt_filename}, epoch finished: {epoch_finished}")
        ckpt = checkpoint.load_data(ckpt_filename)
        params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")
        params = None
    return params

def generate_crystal(spacegroup, elements, temperature, seed, params, transformer, args):
    print(f"Generating with spacegroup={spacegroup}, elements={elements}, temperature={temperature}")
    top_p = 1
    n_sample = 1
    elements = elements.split()
    if elements:
        idx = [element_dict[e] for e in elements]
        atom_mask = [1] + [1 if a in idx else 0 for a in range(1, args.atom_types)]
        atom_mask = jnp.array(atom_mask)
    else:
        atom_mask = jnp.zeros((args.atom_types), dtype=int)
    
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    start_time = time()
    XYZ, A, W, M, L = sample_crystal(
        subkey, 
        transformer, 
        params, 
        args.n_max, 
        n_sample, 
        args.atom_types, 
        args.wyck_types, 
        args.Kx, args.Kl, 
        spacegroup, None, 
        atom_mask, top_p, 
        temperature, 
        temperature, 
        args.use_foriloop)
        
    end_time = time()
    print(f"Execution time: {end_time - start_time}")

    XYZ, A, W, L = map(np.array, (XYZ, A, W, L))
    G = np.array([spacegroup for _ in range(len(L))])
    
    structures = [get_struct_from_lawx(g, l, a, w, xyz) for g, l, a, w, xyz in zip(G, L, A, W, XYZ)]
    structures = [Structure.from_dict(_) for _ in structures]
    atoms_list = [AseAtomsAdaptor().get_atoms(struct) for struct in structures]
    
    return atoms_list

def run_crystalformer(spacegroup, elements, temperature, seed):
    configure_jax()
    with initialize(version_base=None, config_path="./model"):
        args = compose(config_name="config")
    params, transformer = initialize_parameters(args)
    restore_path = "/share/"

    global tempdir
    params = load_checkpoint(restore_path) if params is None else params    
    atoms_list = generate_crystal(spacegroup, elements, temperature, seed, params, transformer, args) 
    outputPath = os.path.join(tempdir, "pred_struct.cif")   
    write(outputPath, atoms_list[-1], format="cif")

    return outputPath

def main():
    global tempdir
    with tempfile.TemporaryDirectory(dir=".") as tempdir:
        with gr.Blocks() as app:
            with gr.Tab(label="CrystalFormer"):
                with gr.Row():
                    with gr.Column():
                        spacegroup = gr.Number(label="Spacegroup", value=225)
                        elements = gr.Textbox(label="Elements", value="C")
                    with gr.Column():
                        temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                        seed = gr.Number(label="Seed", value=42)
                
                with gr.Row():
                    generate_btn = gr.Button("Generate Structure")
                    clear_btn = gr.Button("Clear Inputs")
                
                output_file = gr.File(label="CIF File")
                
                generate_btn.click(
                    fn=run_crystalformer,
                    inputs=[spacegroup, elements, temperature, seed],
                    outputs=[output_file]
                )
                
                clear_btn.click(
                    fn=lambda: (225, "C", 1.0, 42),
                    inputs=None,
                    outputs=[spacegroup, elements, temperature, seed]
                )
            
            gr.Markdown(
                """
                # CrystalFormer
                
                Generate crystal structures using CrystalFormer.
                
                ## Instructions:
                1. Enter the spacegroup number
                2. Specify the elements (comma-separated)
                3. Adjust the temperature
                4. Set a random seed (optional)
                5. Click 'Generate Structure' to create the CIF file
                """
            )
        
        # Launch the Gradio interface
        app.launch(share=True)

if __name__ == "__main__":
    main()