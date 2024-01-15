from diffusers import ControlNetModel, AutoencoderKL, DDIMScheduler
from diffusers.utils import export_to_video
import numpy as np
import torch
from diffusers import DiffusionPipeline
import cv2
from PIL import Image
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
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

input_video_path = "/home/ubuntu/sws/recursive/Rerender_A_Video/videos/horserunning.mp4"
input_interval = 5
frames = video_to_frame(
    input_video_path, input_interval)

control_canny_frames = []
control_tile_frames = []
# get canny image
for frame in frames:
    frame=cv2.resize(frame,(512,320))
    control_tile_frames.append(Image.fromarray(frame))
    np_image = cv2.Canny(frame, 50, 100)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)
    control_canny_frames.append(canny_image)
control_frames = [control_tile_frames, control_canny_frames]
# You can use any ControlNet here
#controlnet = ControlNetModel.from_pretrained(
#    "/home/ubuntu/sws/project/diffusers-Rerender-rerender/sd-controlnet-canny").to('cuda')

controlnet = [
            ControlNetModel.from_pretrained("/home/ubuntu/sws/model/controlnet11/control_v11f1e_sd15_tile", torch_dtype=torch.float32),
            ControlNetModel.from_pretrained("/home/ubuntu/sws/model/controlnet11/control_v11p_sd15_canny", torch_dtype=torch.float32),
            ControlNetModel.from_pretrained("/home/ubuntu/sws/model/controlnet11/control_v11f1p_sd15_depth", torch_dtype=torch.float32),
        ]

# You can use any fintuned SD here
pipe = DiffusionPipeline.from_pretrained(
    "/home/ubuntu/sws/model/diffusers_model/Realistic_Vision_V6.0_B1_noVAE", controlnet=controlnet, custom_pipeline='/home/ubuntu/liudong/diffusers-Rerender-rerender/examples/community/rerender_a_video3.py').to('cuda')
#/home/ubuntu/sws/stable-diffusion-v1-5


# -------- freeu block registration
# register_free_upblock2d(pipe, b1=1.1, b2=1.2, s1=1.0, s2=0.2)
# register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.3, s1=1.0, s2=0.2)
# -------- freeu block registration

# Optional: you can download vae-ft-mse-840000-ema-pruned.ckpt to enhance the results
pipe.vae = AutoencoderKL.from_single_file(
    "/home/ubuntu/sws/model/vae-ft-mse-840000-ema-pruned.ckpt",config_file="/home/ubuntu/sws/project/stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml").to('cuda')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#pipe.enable_xformers_memory_efficient_attention()

generator = torch.manual_seed(0)
# frames = [Image.fromarray(frame) for frame in frames]
framesa = []
for frame in frames:
    frame=cv2.resize(frame,(512,320))
    framesa.append(Image.fromarray(frame))
frames = framesa

# mask_control=torch.ones_like(latents)#(latents.shape[0:2], np.uint8)*255
# zero_lines = int(latents.shape[2] * 2 / 3)
# mask_control[:,:, 0:zero_lines, :]=0
#mask_control = None

output_path="/home/ubuntu/liudong/diffusers-Rerender-rerender/video/horse2canny-tail_diffout42.mp4"

output_frames = pipe(
    # "white ancient Greek sculpture, Venus de Milo, light pink and blue background",
    "sand flying,A horse was running in the desert, sand flying between its hooves",#,RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
    frames,
    control_frames,
    num_inference_steps=20,
    strength=0.95,
    controlnet_conditioning_scale=[0.35, 0.5],  # tile, canny
    generator=generator,
    warp_start=0.0,
    warp_end=0.1,
    mask_start=0.5,
    mask_end=0.4,
    bUseMaskControl=False,
    color_preserve = True,
    mask_strength=0.5,
    output_path=output_path,
    control_guidance_start=0,
    control_guidance_end=1,
    negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
    #'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
).frames

export_to_video(
    output_frames,output_path, 10)