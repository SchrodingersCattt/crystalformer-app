import gradio as gr
from gradio_materialviewer import MaterialViewer
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_op import run_crystalformer
from op import run_op, run_op_gpu

current_tempdir = None  # Global variable to store the current temporary directory

def main():
    with gr.Blocks() as app:
        with gr.Tab(label="Quick Start Mode"):
            with gr.Row():
                with gr.Column():
                    qs_spacegroup = gr.Slider(label="Spacegroup", minimum=1, maximum=230, value=225, step=1)
                    qs_elements = gr.Textbox(label="Elements", value="C")
                with gr.Column():
                    qs_temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    qs_seed = gr.Number(label="Seed", value=42)
                    
            with gr.Row():
                qs_generate_btn = gr.Button("Generate Structure")
                qs_clear_btn = gr.Button("Clear Inputs")

            qs_output_file = gr.File(label="CIF File")
            qs_material_viewer = MaterialViewer(height=480, materialFile="", format='cif')

            def generate_and_display_structure_quick(sp, el, temp, sd):
                global current_tempdir
                if current_tempdir:
                    current_tempdir.cleanup()  # Clean up the previous temporary directory
                current_tempdir = tempfile.TemporaryDirectory(dir=".")  # Create a new temporary directory
                cif_file_path = run_crystalformer(sp, el, temp, sd, current_tempdir.name)
                with open(cif_file_path, 'r') as ff:
                    cif_content = "".join(ff.readlines())
                return cif_file_path, MaterialViewer(materialFile=cif_content, format='cif', height=480)

            qs_generate_btn.click(
                fn=generate_and_display_structure_quick,
                inputs=[qs_spacegroup, qs_elements, qs_temperature, qs_seed],
                outputs=[qs_output_file, qs_material_viewer]
            )

            qs_clear_btn.click(
                fn=lambda: (225, "C", 1.0, 42),
                inputs=None,
                outputs=[qs_spacegroup, qs_elements, qs_temperature, qs_seed]
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
                with gr.Column():
                    access_key = gr.Textbox(label="Access Key")
                    project_id = gr.Textbox(label="Project ID")

                with gr.Column():
                    machine_type = gr.Dropdown(label="Machine Type", choices=[
                        "c12_m64_1 * NVIDIA L4",
                        "1 * NVIDIA T4_16g",
                        "1 * NVIDIA V100_32g",
                    ])
                    image = gr.Textbox(label="Image", value="registry.dp.tech/dptech/prod-19853/crystal-former:0.0.2")

            with gr.Row():
                generateGPU_btn = gr.Button("Generate Structure on GPU machines")
                clear_btn = gr.Button("Clear Inputs")

            output_file = gr.File(label="CIF File")
            error_message = gr.Markdown(label="Error", visible=False)
            material_viewer = MaterialViewer(height=480, materialFile="", format='cif')

            def generate_and_display_structure_gpu(sp, el, wy, temp, sd, T1, ns, ak, pid, mt, im):
                global current_tempdir
                if current_tempdir:
                    current_tempdir.cleanup()
                current_tempdir = tempfile.TemporaryDirectory(dir=".")
                result = run_op_gpu(sp, el, wy, temp, sd, T1, ns, ak, pid, mt, im, current_tempdir.name)
                
                if "error" in result:
                    error_message_content = f"**Error:** {result['error']}"
                    return None, gr.update(visible=True, value=error_message_content), None
                
                cif_file_path = result
                with open(cif_file_path, 'r') as ff:
                    cif_content = "".join(ff.readlines())
                return cif_file_path, gr.update(visible=False), MaterialViewer(materialFile=cif_content, format='cif', height=480)

            generateGPU_btn.click(
                fn=generate_and_display_structure_gpu,
                inputs=[spacegroup, elements, wyckoff, temperature, seed, T1, nsweeps, access_key, project_id, machine_type, image],
                outputs=[output_file, error_message, material_viewer]
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

    app.launch(server_name='0.0.0.0', server_port=50001)

if __name__ == "__main__":
    main()