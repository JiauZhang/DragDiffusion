import gradio as gr
from dragdiffusion import DragDiffusion

device = 'cuda'
cache_dir = '~/.cache/hf'
model = DragDiffusion(device, cache_dir)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Box():
            prompt = gr.Textbox(
                show_label=False, lines=1, placeholder="Enter your prompt",
            )
            with gr.Row():
                seed = gr.Number(
                    value=521, label='Seed', precision=0,
                    minimum=0, maximum=2147483647, interactive=True,
                )
                gen_btn = gr.Button(
                    value='generate image', scale=1,
                )
        image = gr.Image(type='numpy', shape=(512, 512))
        gen_btn.click(
            model.generate_image, inputs=[prompt, seed], outputs=[image],
        )

demo.launch()
