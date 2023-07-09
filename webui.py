import numpy as np
import gradio as gr
from dragdiffusion import DragDiffusion

device = 'cuda'
cache_dir = './.cache/hf'
model = DragDiffusion(device, cache_dir)
add_point, points = 0, []
point_color = [(255, 0, 0, 255), (0, 0, 255, 255)]
MASK_VALUE = 125
dragging = False

def add_point_cb():
    global add_point
    add_point += 2

def reset_point_cb():
    global points
    points = []

def start_cb(image):
    global points, steps, dragging
    if dragging: return
    dragging = True
    while (dragging):
        image = model.step(points)
        yield image

def stop_cb():
    global dragging
    dragging = False
    print('stop dragging...')

def drag_box():
    add_btn = gr.Button(value='add point', scale=1, min_width=20)
    add_btn.click(add_point_cb)
    reset_btn = gr.Button(value='reset point', scale=1, min_width=20)
    reset_btn.click(reset_point_cb)
    start_btn = gr.Button(value='start', scale=1, min_width=20)
    stop_btn = gr.Button(value='stop', scale=1, min_width=20)
    stop_btn.click(stop_cb, queue=False)
    return start_btn

# seed: 88937258, 893726, 893727, 853378485
# prompt: ad for a burger with ham, patty, egg and lettuce leaf, freshly cooked with light smoke,
# burger in the center of the frame in bright light, background flame fire darker, lens 50 mm f2.8
def gen_box():
    with gr.Box():
        prompt = gr.Textbox(
            label='Prompt', lines=1, value="ad for a burger with ham, patty, egg and lettuce leaf, "
            + "freshly cooked with light smoke, burger in the center of the frame in bright light, "
            + "background flame fire darker, lens 50 mm f2.8",
        )
        with gr.Row():
            seed = gr.Number(
                value=88937258, label='Seed', precision=0,
                minimum=0, maximum=2147483647, interactive=True,
            )
            steps = gr.Number(
                value=50, label='Steps', precision=0,
                minimum=1, maximum=1000, interactive=True,
            )
            gen_btn = gr.Button(value='generate image', scale=1)
        with gr.Row():
            cfg_scale = gr.Slider(
                value=1, label='CFG Scale', precision=0,
                minimum=1, maximum=100, interactive=True,
            )
            time_step = gr.Number(
                value=40, label='Time Step', precision=0,
                minimum=1, maximum=50, interactive=True,
            )
        with gr.Row():
            start_btn = drag_box()
    return (prompt, seed, steps, gen_btn, cfg_scale, time_step,
        start_btn,
    )

def select_mask(image):
    mask_ = image[..., -1]
    mask_[mask_ == MASK_VALUE] = 255
    image[..., -1] = mask_
    return image

def select_image(mask, image):
    mask_ = mask['mask'][..., 0]
    mask_[mask_ == 0] = MASK_VALUE
    image[..., -1] = mask_
    return image

def image_mask_box():
    with gr.Box():
        with gr.Tab('image') as image_tab:
            image = gr.Image(
                type='numpy',value=np.ones((512, 512, 3)), image_mode='RGBA',
            ).style(height=512, width=512)
        with gr.Tab('mask') as mask_tab:
            mask = gr.ImageMask().style(height=512, width=512)
        image_tab.select(select_image, inputs=[mask, image], outputs=[image])
        mask_tab.select(select_mask, inputs=[image], outputs=[mask])
    return image, mask

def draw_point(image, x, y, color, radius=3):
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
        prompt, seed, steps, gen_btn, cfg_scale, time_step, start_btn = gen_box()
        image, mask = image_mask_box()
    gen_btn.click(
        model.generate_image, inputs=[prompt, seed, steps, cfg_scale, time_step], outputs=[image],
    )
    image.select(select_point, inputs=[image], outputs=[image])
    start_btn.click(start_cb, inputs=[image], outputs=[image])

demo.launch(enable_queue=True)
