import numpy as np
import gradio as gr
from dragdiffusion import DragDiffusion

device = 'cuda'
cache_dir = './.cache/hf'
model = DragDiffusion(device, cache_dir)
add_point, points = 0, []
point_color = [(255, 0, 0), (0, 0, 255)]

def add_point_cb():
    global add_point
    add_point += 2

def reset_point_cb():
    global points
    points = []

def drag_box():
    add_btn = gr.Button(value='add point', scale=1, min_width=20)
    add_btn.click(add_point_cb)
    reset_btn = gr.Button(value='reset point', scale=1, min_width=20)
    reset_btn.click(reset_point_cb)
    start_btn = gr.Button(value='start', scale=1, min_width=20)
    stop_btn = gr.Button(value='stop', scale=1, min_width=20)

def gen_box():
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
            gen_btn = gr.Button(value='generate image', scale=1)
        with gr.Row():
            drag_box()
    return prompt, seed, steps, gen_btn

def draw_point(image, x, y, color, radius=2):
    x_start, x_end = max(0, x - radius), min(512, x + radius)
    y_start, y_end = max(0, y - radius), min(512, y + radius)
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            image[y, x] = color
    return image

def select_point(image, event: gr.SelectData):
    global add_point, points
    if add_point <= 0: return image
    ix, iy = event.index
    image = draw_point(image, ix, iy, point_color[add_point % 2])
    points.append(np.array([ix, iy]))
    print(points)
    add_point -= 1
    return image

with gr.Blocks() as demo:
    with gr.Row():
        prompt, seed, steps, gen_btn = gen_box()
        image = gr.Image(type='numpy', shape=(512, 512), value=np.ones((512, 512, 3)))
        gen_btn.click(
            model.generate_image, inputs=[prompt, seed, steps], outputs=[image],
        )
        image.select(select_point, inputs=[image], outputs=[image])

demo.launch()
