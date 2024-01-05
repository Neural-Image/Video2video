from diffusers import ControlNetModel, AutoencoderKL, DDIMScheduler
from diffusers.utils import export_to_video
import numpy as np
import torch
from diffusers import DiffusionPipeline
import cv2
from PIL import Image

import sys
gmflow_dir = "/home/ubuntu/sws/recursive/Rerender_A_Video/deps/gmflow/"
sys.path.insert(0, gmflow_dir)


def video_to_frame(video_path: str, interval: int):
    vidcap = cv2.VideoCapture(video_path)
    success = True

    count = 0
    res = []
    while success:
        count += 1
        success, image = vidcap.read()
        if count % interval != 1:
            continue
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res.append(image)

    vidcap.release()
    return res

input_video_path = "/home/ubuntu/sws/recursive/Rerender_A_Video/videos/pexels-cottonbro-studio-6649832-960x506-25fps.mp4" #"/home/ubuntu/sws/recursive/Rerender_A_Video/videos/horserunning.mp4"
input_interval = 10
frames = video_to_frame(
    input_video_path, input_interval)

control_frames = []
# get canny image
for frame in frames:
    frame=cv2.resize(frame,(512,320))
    np_image = cv2.Canny(frame, 50, 100)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)
    control_frames.append(canny_image)

# You can use any ControlNet here
controlnet = ControlNetModel.from_pretrained(
    "/home/ubuntu/sws/model/controlnet11/control_v11p_sd15_canny").to('cuda')

# You can use any fintuned SD here
pipe = DiffusionPipeline.from_pretrained(
    "/home/ubuntu/sws/stable-diffusion-v1-5", controlnet=controlnet, custom_pipeline='/home/ubuntu/sws/project/diffusers-Rerender-rerender/examples/community/rerender_a_video1.py').to('cuda')

# Optional: you can download vae-ft-mse-840000-ema-pruned.ckpt to enhance the results
# pipe.vae = AutoencoderKL.from_single_file(
#     "/home/ubuntu/sws/model/vae-ft-mse-840000-ema-pruned.ckpt").to('cuda')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(0)
#frames = [Image.fromarray(frame) for frame in frames]
framesa = []
for frame in frames:
    frame=cv2.resize(frame,(512,320))
    framesa.append(Image.fromarray(frame))
frames = framesa

output_frames = pipe(
    "white ancient Greek sculpture, Venus de Milo, light pink and blue background",
    # "A horse was running in the desert, sand flying between its hooves",

    frames,
    control_frames,
    num_inference_steps=20,
    strength=0.75,
    controlnet_conditioning_scale=0.7,
    generator=generator,
    warp_start=0.0,
    warp_end=0.1,
    mask_start=0.5,
    mask_end=0.8,
    mask_strength=0.5,
    negative_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
).frames

export_to_video(
    output_frames, "/home/ubuntu/sws/recursive/Rerender_A_Video/videos/horserunning_diffout11.mp4", 5)