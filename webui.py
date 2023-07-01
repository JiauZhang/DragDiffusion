import gradio as gr
from dragdiffusion import DragDiffusion

device = 'cuda'
cache_dir = './.cache/hf'
model = DragDiffusion(device, cache_dir)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Box():
            prompt = gr.Textbox(
                label='Prompt', lines=1, value="Chinese Panda",
            )
            with gr.Row():
                seed = gr.Number(
                    value=19491001, label='Seed', precision=0,
                    minimum=0, maximum=2147483647, interactive=True,
                )
                steps = gr.Number(
                    value=50, label='Steps', precision=0,
                    minimum=1, maximum=1000, interactive=True,
                )
                gen_btn = gr.Button(
                    value='generate image', scale=1,
                )
        image = gr.Image(type='numpy', shape=(512, 512))
        gen_btn.click(
            model.generate_image, inputs=[prompt, seed, steps], outputs=[image],
        )

demo.launch()
