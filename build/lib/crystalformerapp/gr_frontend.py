import gradio as gr
from gradio_materialviewer import MaterialViewer
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_op import run_crystalformer
from op import run_op, run_op_gpu

def main():
    with tempfile.TemporaryDirectory(dir=".") as tempdir:
        with gr.Blocks() as app:
            with gr.Tab(label="Quick Start Mode"):
                with gr.Row():
                    with gr.Column():
                        spacegroup = gr.Slider(label="Spacegroup", minimum=1, maximum=230, value=225, step=1)
                        elements = gr.Textbox(label="Elements", value="C")
                    with gr.Column():
                        temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                        seed = gr.Number(label="Seed", value=42)
                        
                with gr.Row():
                    generate_btn = gr.Button("Generate Structure")
                    clear_btn = gr.Button("Clear Inputs")

                output_file = gr.File(label="CIF File")
                material_viewer = MaterialViewer(height=480, materialFile="", format='cif')

                def generate_and_display_structure(sp, el, temp, sd):
                    cif_file_path = run_crystalformer(sp, el, temp, sd, tempdir)
                    with open(cif_file_path, 'r') as ff:
                        cif_content = "".join(ff.readlines())
                    return cif_file_path, MaterialViewer(materialFile=cif_content, format='cif', height=480)

                generate_btn.click(
                    fn=generate_and_display_structure,
                    inputs=[spacegroup, elements, temperature, seed],
                    outputs=[output_file, material_viewer]
                )

                clear_btn.click(
                    fn=lambda: (225, "C", 1.0, 42),
                    inputs=None,
                    outputs=[spacegroup, elements, temperature, seed]
                )

                gr.Markdown("""
                    # Quick Start Mode

                    Generate crystal structures with Quick Start Mode.

                    ## Instructions:
                    1. Enter the spacegroup number
                    2. Specify the elements (comma-separated)
                    3. Adjust the temperature
                    4. Set a random seed (optional)
                    5. Click 'Generate Structure' to create the CIF file
                """)

            with gr.Tab(label="Research Mode"):
                with gr.Row():
                    with gr.Column():
                        spacegroup = gr.Slider(label="Spacegroup", minimum=1, maximum=230, value=225, step=1)
                        elements = gr.Textbox(label="Elements", value="C")
                        wyckoff = gr.Textbox(label="Wyckoff", value="a")
                    with gr.Column():
                        seed = gr.Number(label="Seed", value=42)
                        temperature = gr.Slider(label="Temperature", minimum=0.5, maximum=1.5, value=1.0, step=0.1)
                        T1 = gr.Slider(label="T1", minimum=100, maximum=100000000, value=100, step=100)
                        nsweeps = gr.Slider(label="nsweeps", minimum=0, maximum=20, value=10, step=1)
                with gr.Row():
                        access_key = gr.Textbox(label="Access Key")
                        project_id = gr.Textbox(label="Project ID")
                        machine_type = gr.Dropdown(label="Machine Type", choices=[
                            "1 * NVIDIA T4_16g",
                            "1 * NVIDIA V100_32g",
                            "c12_m64_1 * NVIDIA L4",
                        ])

                with gr.Row():
                    generateWeb_btn = gr.Button("Generate Structure")
                    generateGPU_btn = gr.Button("Generate Structure on GPU machines")
                    clear_btn = gr.Button("Clear Inputs")

                output_file = gr.File(label="CIF File")
                material_viewer = MaterialViewer(height=480, materialFile="", format='cif')

                def generate_and_display_structure_web(sp, el, wy, temp, sd, T1, ns):
                    cif_file_path = run_op(sp, el, wy, temp, sd, T1, ns, tempdir)
                    with open(cif_file_path, 'r') as ff:
                        cif_content = "".join(ff.readlines())
                    return cif_file_path, MaterialViewer(materialFile=cif_content, format='cif', height=480)

                generateWeb_btn.click(
                    fn=generate_and_display_structure_web,
                    inputs=[spacegroup, elements, wyckoff, temperature, seed, T1, nsweeps],
                    outputs=[output_file, material_viewer]
                )

                def generate_and_display_structure_gpu(sp, el, wy, temp, sd, T1, ns, ak, pid, mt):
                    cif_file_path = run_op_gpu(sp, el, wy, temp, sd, T1, ns, ak, pid, mt, tempdir)
                    with open(cif_file_path, 'r') as ff:
                        cif_content = "".join(ff.readlines())
                    return cif_file_path, MaterialViewer(materialFile=cif_content, format='cif', height=480)

                generateGPU_btn.click(
                    fn=generate_and_display_structure_gpu,
                    inputs=[spacegroup, elements, wyckoff, temperature, seed, T1, nsweeps, access_key, project_id, machine_type],
                    outputs=[output_file, material_viewer]
                )

                clear_btn.click(
                    fn=lambda: (225, "C", 1.0, 42),
                    inputs=None,
                    outputs=[spacegroup, elements, temperature, seed]
                )

                gr.Markdown("""
                    # Research Mode

                    Generate crystal structures with Research Mode.

                    ## Instructions:
                    - **seed**: random seed to sample the crystal structure
                    - **spacegroup**: control the space group of generated crystals
                    - **temperature**: modifies the probability distribution
                    - **T1**: the temperature of sampling the first atom type
                    - **elements**: control the elements in the generating process. Note that you need to enter the elements separated by spaces, i.e., Ba Ti O, if the elements string is none, the model will not limit the elements
                    - **wyckoff**: control the Wyckoff letters in the generation. Note that you need to enter the Wyckoff letters separated by spaces, i.e., a c, if the Wyckoff is none, the model will not limit the wyckoff letter.
                    - **nsweeps**: control the steps of mcmc to refine the generated structures
                """)

        app.launch(share=True)

if __name__ == "__main__":
    main()