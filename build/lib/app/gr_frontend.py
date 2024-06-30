import gradio as gr
from gradio_materialviewer import MaterialViewer
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crystalformerapp.simple_op import run_crystalformer
from crystalformerapp.op import run_op, run_op_gpu

current_tempdir = None  # Global variable to store the current temporary directory


def main():
    with gr.Blocks() as app:
        with gr.Tab(label="CIF tools"):
            pass
        
        with gr.Tab(label="Searching in Database"):
            pass
            
        with gr.Tab(label="Crystal Former"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    with gr.Row():
                        spacegroup = gr.Slider(label="Spacegroup Number", minimum=1, maximum=230, value=1, step=1)
                        elements = gr.Textbox(label="Elements, splitted by a space", value="")
                        wyckoff = gr.Textbox(label="Wyckoff, splitted by a space", value="")
                        nsample = gr.Slider(label="nsample", minimum=1, maximum=4, value=1, step=1)
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=42)
                        temperature = gr.Slider(label="Temperature", minimum=0.5, maximum=1.5, value=1.0, step=0.1)
                        T1 = gr.Slider(label="T1", minimum=100, maximum=100000000, value=100, step=100)
                        nsweeps = gr.Slider(label="nsweeps", minimum=0, maximum=20, value=10, step=1)

                with gr.Column(scale=1):
                    output_files = gr.File(label="Download CIF File")
                    error_message = gr.Markdown(label="Error", visible=False)
                    
            with gr.Row(visible=False) as gpu_parameters:
                with gr.Column():
                    access_key = gr.Textbox(label="Access Key")
                    project_id = gr.Textbox(label="Project ID")

                with gr.Column():
                    machine_type = gr.Dropdown(label="Machine Type",value="c12_m64_1 * NVIDIA L4",
                                              choices=[
                                               "c12_m64_1 * NVIDIA L4",
                                               "1 * NVIDIA T4_16g",
                                               "1 * NVIDIA V100_32g",
                                               "1 * NVIDIA GPU_24g",
                                               "c6_m64_1 * NVIDIA 3090",
                                               "c6_m60_1 * NVIDIA 4090",
                                               "c4_m15_1 * NVIDIA T4"
                                               
                    ])
                    image = gr.Textbox(label="Image", value="registry.dp.tech/dptech/prod-19853/crystal-former:0.0.8")
                
                generateGPU_btn_run = gr.Button("Submit Your Job to Bohrium")
            
            with gr.Row():
                quickMode_btn = gr.Button("Generate Structure in QuickStart Mode")
                generateGPU_btn = gr.Button("Generate Structure on GPU machines (Bohrium access required)")
                #clear_btn = gr.Button("Clear Inputs")
            
            with gr.Row():
                material_viewer0 = MaterialViewer(height=360, materialFile="", format='cif')
                material_viewer1 = MaterialViewer(height=360, materialFile="", format='cif')    
                material_viewer2 = MaterialViewer(height=360, materialFile="", format='cif') 
                material_viewer3 = MaterialViewer(height=360, materialFile="", format='cif')               

            def generate_and_display_structure_gpu(sp, el, wy, temp, sd, T1, ns, np, ak, pid, mt, im):
                global current_tempdir
                if current_tempdir:
                    current_tempdir.cleanup()
                current_tempdir = tempfile.TemporaryDirectory(dir=".")
                result = run_op_gpu(sp, el, wy, temp, sd, T1, ns, np, ak, pid, mt, im, current_tempdir.name)
                
                if isinstance(result, dict) and "error" in result:
                    error_message_content = f"**Error:** {result['error']}"
                    return None, gr.update(visible=True, value=error_message_content), None, None, None, None
                
                cif_file_paths = result
                with open(cif_file_paths[0], 'r') as ff0, \
                    open(cif_file_paths[1], 'r') as ff1, \
                    open(cif_file_paths[2], 'r') as ff2, \
                    open(cif_file_paths[3], 'r') as ff3:
                    cif_content0 = "".join(ff0.readlines())
                    cif_content1 = "".join(ff1.readlines())
                    cif_content2 = "".join(ff2.readlines())
                    cif_content3 = "".join(ff3.readlines())

                return  (cif_file_paths, 
                        gr.update(visible=False),
                        MaterialViewer(materialFile=cif_content0, format='cif', height=360),
                        MaterialViewer(materialFile=cif_content1, format='cif', height=360),
                        MaterialViewer(materialFile=cif_content2, format='cif', height=360),
                        MaterialViewer(materialFile=cif_content3, format='cif', height=360))
            
            def generate_and_display_structure_quickstart(sp, el, wy, temp, sd, T1, ns, np):
                global current_tempdir
                if current_tempdir:
                    current_tempdir.cleanup()
                current_tempdir = tempfile.TemporaryDirectory(dir=".")
                result = run_op(sp, el, wy, temp, sd, T1, ns, np, current_tempdir.name)
                
                if isinstance(result, dict) and "error" in result:
                    error_message_content = f"**Error:** {result['error']}"
                    return None, gr.update(visible=True, value=error_message_content), None, None, None, None
                
                cif_file_paths = result
                with open(cif_file_paths[0], 'r') as ff0, \
                    open(cif_file_paths[1], 'r') as ff1, \
                    open(cif_file_paths[2], 'r') as ff2, \
                    open(cif_file_paths[3], 'r') as ff3:
                    cif_content0 = "".join(ff0.readlines())
                    cif_content1 = "".join(ff1.readlines())
                    cif_content2 = "".join(ff2.readlines())
                    cif_content3 = "".join(ff3.readlines())

                return (cif_file_paths, 
                        gr.update(visible=False), 
                        MaterialViewer(materialFile=cif_content0, format='cif', height=360), 
                        MaterialViewer(materialFile=cif_content1, format='cif', height=360), 
                        MaterialViewer(materialFile=cif_content2, format='cif', height=360), 
                        MaterialViewer(materialFile=cif_content3, format='cif', height=360))

            quickMode_btn.click(
                fn=generate_and_display_structure_quickstart,
                inputs=[spacegroup, elements, wyckoff, temperature, seed, T1, nsweeps, nsample],
                outputs=[output_files, error_message, material_viewer0, material_viewer1, material_viewer2, material_viewer3]
            )
            
            generateGPU_btn.click(
                fn=lambda: gr.update(visible=True),
                inputs=[],
                outputs=[gpu_parameters]
            )
            
            generateGPU_btn_run.click(
                fn=generate_and_display_structure_gpu,
                inputs=[spacegroup, elements, wyckoff, temperature, seed, T1, nsweeps, nsample, access_key, project_id, machine_type, image],
                outputs=[output_files, error_message, material_viewer0, material_viewer1, material_viewer2, material_viewer3],
                show_progress=True
            )
            '''
            clear_btn.click(
                fn=lambda: (225, "C", "a", 4, 42, 1.0, 100, 10, "", "", "c12_m64_1 * NVIDIA L4", "registry.dp.tech/dptech/prod-19853/crystal-former:0.0.7"),
                inputs=None,
                outputs=[spacegroup, elements, wyckoff, nsample, seed, temperature, T1, nsweeps, access_key, project_id, machine_type, image]
            )'''

            gr.Markdown("""
                # Crystal Former APP

                Generate new crystal structures with `CrystalFormer`.
                
                For more details, please check [Github](https://github.com/deepmodeling/CrystalFormer) and [paper](https://arxiv.org/abs/2403.15734).
                                
                ## Instructions:
                - **seed**: random seed to sample the crystal structure
                - **spacegroup**: control the space group of generated crystals
                - **temperature**: modifies the probability distribution
                - **T1**: the temperature of sampling the first atom type
                - **elements**: control the elements in the generating process. Note that you need to enter the elements separated by spaces, i.e., Ba Ti O, if the elements string is none, the model will not limit the elements
                - **wyckoff**: control the Wyckoff letters in the generation. Note that you need to enter the Wyckoff letters separated by spaces, i.e., a c, if the Wyckoff is none, the model will not limit the wyckoff letter.
                - **nsample**: control the batch size of every generation.
                - **nsweeps**: control the steps of mcmc to refine the generated structures
            """)

    app.launch(server_name='0.0.0.0',server_port=50001, width="100%")

if __name__ == "__main__":
    main()