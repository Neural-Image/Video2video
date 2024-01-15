from diffusers import ControlNetModel, AutoencoderKL, DDIMScheduler
from diffusers.utils import export_to_video
import numpy as np
import torch
from diffusers import DiffusionPipeline,StableDiffusionControlNetInpaintPipeline
from controlnet_aux import LineartDetector

import cv2
from PIL import Image
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
import sys
import os
gmflow_dir = "/home/ubuntu/sws/recursive/Rerender_A_Video/deps/gmflow/"
sys.path.insert(0, gmflow_dir)

# new_frame_size=(448,896)#60
# new_frame_size=(320,512)#45
new_frame_size=(544,960) #70
# new_frame_size=(448,768)# 56
output_path="/home/ubuntu/liudong/diffusers-Rerender-rerender/video/sce52inpaint-tail_diffout27.mp4"

output_lineart_path=os.path.join("/home/ubuntu/sws/recursive/Rerender_A_Video/videos/lineart",output_path.split("/")[-1].split(".")[0])
if not os.path.exists(output_lineart_path):
    os.makedirs(output_lineart_path)

horse_mask_path="/home/ubuntu/sws/data/horse_mask/twohorsemask_ground2"



lineart_processor = LineartDetector.from_pretrained(pretrained_model_or_path="/home/ubuntu/sws/model/diffusers_model/Annotators")



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

def make_inpaint_condition_horse_mask(image,horse_mask):
    image = np.array(image).astype(np.float32) / 255.0
    # image_mask = np.array(image_mask).astype(np.float32) / 255.0
    # image_mask=np.zeros((new_frame_size[0],new_frame_size[1],3),dtype=np.uint8)
    image_mask=horse_mask
    # cv2.imwrite("/home/ubuntu/sws/recursive/Rerender_A_Video/videos/1.jpg",image_mask)
    # image_mask[(new_frame_size[0]-60):new_frame_size[0],:] = [255,255,255]
    # cv2.imwrite("/home/ubuntu/sws/recursive/Rerender_A_Video/videos/2.jpg",image_mask)
    image_mask=image_mask.astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    vis_image=image*255
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/home/ubuntu/sws/recursive/Rerender_A_Video/videos/3.jpg",vis_image)
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)

    image = torch.from_numpy(image)
    return image

def make_inpaint_condition(image):
    image = np.array(image).astype(np.float32) / 255.0
    # image_mask = np.array(image_mask).astype(np.float32) / 255.0
    image_mask=np.zeros((new_frame_size[0],new_frame_size[1],3),dtype=np.uint8)
    cv2.imwrite("/home/ubuntu/sws/recursive/Rerender_A_Video/videos/1.jpg",image_mask)
    image_mask[(new_frame_size[0]-60):new_frame_size[0],:] = [255,255,255]
    cv2.imwrite("/home/ubuntu/sws/recursive/Rerender_A_Video/videos/2.jpg",image_mask)
    image_mask=image_mask.astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    vis_image=image*255
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/home/ubuntu/sws/recursive/Rerender_A_Video/videos/3.jpg",vis_image)
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)

    image = torch.from_numpy(image)
    return image

def to_frame(video_path: str):

    res = []
    framelist=os.listdir(video_path)
    framelist.sort()
    for image in framelist:
        image=cv2.imread(os.path.join(video_path,image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res.append(image)

    return res

# input_video_path = "/home/ubuntu/sws/data/testsce5/output2.mp4"
# input_interval = 2
# frames = video_to_frame(
#     input_video_path, input_interval)

input_frame_path="/home/ubuntu/sws/data/testsce5/merged" #Scene5.0000"
# input_frame_path="/home/ubuntu/liudong/diffusers-Rerender-rerender/video/moved_imgs"

frames = to_frame(
    input_frame_path)
horse_mask = to_frame(
    horse_mask_path)



control_inpaint_frames = []
control_tile_frames = []
control_lineart_frames = []
# get canny image
for idx,horse_mask_idx in enumerate (horse_mask):
    frame=frames[idx]
# for idx,frame in enumerate (frames):
#     horse_mask_idx=horse_mask[idx]

    horse_mask_idx=cv2.resize(horse_mask_idx,(new_frame_size[1],new_frame_size[0]))
    frame=cv2.resize(frame,(new_frame_size[1],new_frame_size[0]))
    control_tile_frames.append(Image.fromarray(frame))

    lineart_image = lineart_processor(Image.fromarray(frame))
    lineart_image=lineart_image.resize((new_frame_size[1],new_frame_size[0]))
    lineart_image.save(os.path.join(output_lineart_path,str(idx)+".jpg"))
    control_lineart_frames.append(lineart_image)


    # inpaint_frames=make_inpaint_condition(frame)
    inpaint_frames=make_inpaint_condition_horse_mask(frame,horse_mask_idx)
    # np_image = cv2.Canny(frame, 50, 100)
    # np_image = np_image[:, :, None]
    # np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    # canny_image = Image.fromarray(np_image)
    control_inpaint_frames.append(inpaint_frames)
control_frames = [control_tile_frames,control_lineart_frames, control_inpaint_frames]
# You can use any ControlNet here
#controlnet = ControlNetModel.from_pretrained(
#    "/home/ubuntu/sws/project/diffusers-Rerender-rerender/sd-controlnet-canny").to('cuda')

controlnet = [
            ControlNetModel.from_pretrained("/home/ubuntu/sws/model/controlnet11/control_v11f1e_sd15_tile", torch_dtype=torch.float32),
            ControlNetModel.from_pretrained("/home/ubuntu/sws/model/controlnet11/control_v11p_sd15_lineart", torch_dtype=torch.float32),#control_v11f1e_sd15_tile
            ControlNetModel.from_pretrained("/home/ubuntu/sws/model/controlnet11/control_v11p_sd15_inpaint", torch_dtype=torch.float32),
        ]

# You can use any fintuned SD here
pipe = DiffusionPipeline.from_pretrained(
    "/home/ubuntu/sws/model/diffusers_model/Realistic_Vision_V5.1_noVAE", controlnet=controlnet, custom_pipeline='/home/ubuntu/liudong/diffusers-Rerender-rerender/examples/community/rerender_a_video.py').to('cuda')
#/home/ubuntu/sws/stable-diffusion-v1-5  /home/ubuntu/sws/model/diffusers_model/Realistic_Vision_V5.1_noVAE  Realistic_Vision_V6.0_B1_noVAE 

# pipe_refine = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#  "/home/ubuntu/liudong/newrisvol/app/_data/pretrained/epiCRealism/", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16).to('cuda')

# pipe_refine.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_refine.scheduler.config)

# -------- freeu block registration
# register_free_upblock2d(pipe, b1=1.1, b2=1.2, s1=1.0, s2=0.2)
# register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.3, s1=1.0, s2=0.2)
# -------- freeu block registration

# Optional: you can download vae-ft-mse-840000-ema-pruned.ckpt to enhance the results
pipe.vae = AutoencoderKL.from_single_file(
    "/home/ubuntu/sws/model/vae-ft-mse-840000-ema-pruned.ckpt",config_file="/home/ubuntu/sws/project/stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml").to('cuda')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()

generator = torch.manual_seed(0)
# frames = [Image.fromarray(frame) for frame in frames]
framesa = []
for frame in frames:
    frame=cv2.resize(frame,(new_frame_size[1],new_frame_size[0]))
    framesa.append(Image.fromarray(frame))
frames = framesa

# mask_control=torch.ones_like(latents)#(latents.shape[0:2], np.uint8)*255
# zero_lines = int(latents.shape[2] * 2 / 3)
# mask_control[:,:, 0:zero_lines, :]=0
#mask_control = None



output_frames = pipe(
    # "white ancient Greek sculpture, Venus de Milo, light pink and blue background",
    "There are mountains in the desert and white clouds in the sky,sand flying,two white horses were running in the desert, sand flying between their hooves",#sand flying,two horse were running in the desert, sand flying between their hooves",#,RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
    frames,
    control_frames,
    num_inference_steps=20,
    strength=0.95,
    controlnet_conditioning_scale=[0.35,0.6, 0.5],  # tile, lineart,inpatint,
    generator=generator,
    warp_start=0.0,
    warp_end=0.1,
    mask_start=0.5,
    mask_end=0.4,
    bUseMaskControl=True,
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