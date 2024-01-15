import numpy as np
import cv2
import os
from PIL import Image

from diffusers.utils import export_to_video


def add_ground_mask(mask_path,add_ground_mask_path):
    if not os.path.exists(add_ground_mask_path):
        os.mkdir(add_ground_mask_path)

    framelist=os.listdir(mask_path)
    framelist.sort()
    for image_name in framelist:
        
        image=cv2.imread(os.path.join(mask_path,image_name))
        image[860:,:,:]=[255,255,255]


        image=cv2.imwrite(os.path.join(add_ground_mask_path,image_name),image)

def mask2img(mask_path,img_path,out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    framelist=os.listdir(img_path)
    framelist.sort()
    for image_name in framelist:
        image=Image.open(os.path.join(img_path,image_name))
        image=image.resize((1920,1080))
        mask=Image.open(os.path.join(mask_path,image_name))
        # mask=Image.open(os.path.join(mask_path,"Scene5."+"%04d"%(int(image_name.split(".")[0][-4:])+3)+".png"))


        mask = mask.convert('L')
        transparent_im = Image.new('RGBA', image.size, (0, 0, 0,0))
        transparent_im.paste(image, (0, 0), mask)
        transparent_im.save(os.path.join(out_path,image_name))


def mask2img2(mask_path,img_path,out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    framelist=os.listdir(mask_path)
    framelist.sort()
    for image_name in framelist:
        mask=Image.open(os.path.join(mask_path,image_name))
        image=Image.open(os.path.join(img_path,"0000"+image_name.split(".")[1]+".png"))


        mask = mask.convert('L')
        transparent_im = Image.new('RGBA', image.size, (0, 0, 0,0))
        transparent_im.paste(image, (0, 0), mask)
        transparent_im.save(os.path.join(out_path,image_name))

def blend_img(bg_path,horse1_path,horse2_path,blend_out_path):
    if not os.path.exists(blend_out_path):
        os.mkdir(blend_out_path)
    framelist=os.listdir(bg_path)
    framelist.sort()
    for image_name in framelist:
        bg=Image.open(os.path.join(bg_path,image_name))
        horse1=Image.open(os.path.join(horse1_path,"Scene5."+"%04d"%(int(image_name.split(".")[0][-4:]))+".png"))
        horse1=horse1.resize((bg.width,bg.height))
        horse2=Image.open(os.path.join(horse2_path,"Scene5."+"%04d"%(int(image_name.split(".")[0][-4:]))+".png"))
        horse2=horse2.resize((bg.width,bg.height))
        
        bg.paste(horse2, (0,0,bg.width,bg.height),horse2)
        bg.paste(horse1, (0,0,bg.width,bg.height),horse1)
        bg.save(os.path.join(blend_out_path,image_name))
    
def blend_img_1(bg_path,horse1_path,blend_out_path):
    if not os.path.exists(blend_out_path):
        os.mkdir(blend_out_path)
    framelist=os.listdir(bg_path)
    framelist.sort()
    for image_name in framelist:
        bg=Image.open(os.path.join(bg_path,image_name))
        horse1=Image.open(os.path.join(horse1_path,"Scene5."+"%04d"%(int(image_name.split(".")[0][-4:]))+".png"))
        horse1=horse1.resize((bg.width,bg.height))
        # horse2=Image.open(os.path.join(horse2_path,"Scene5."+"%04d"%(int(image_name.split(".")[0][-4:]))+".png"))
        # horse2=horse2.resize((bg.width,bg.height))
        
        # bg.paste(horse2, (0,0,bg.width,bg.height),horse2)
        bg.paste(horse1, (0,0,bg.width,bg.height),horse1)
        bg.save(os.path.join(blend_out_path,image_name))

def horse12_mask_add(horse1_path,horse2_path,horse12_path):
    if not os.path.exists(horse12_path):
        os.mkdir(horse12_path)
    framelist=os.listdir(horse1_path)
    framelist.sort()
    for image_name in framelist:
        horse1=Image.open(os.path.join(horse1_path,image_name))
        horse2=Image.open(os.path.join(horse2_path,image_name))
      
        horse1.paste(horse2, (0,0,horse1.width,horse1.height),horse2)
        # blend=Image.alpha_composite(horse1,horse2)
        horse1.save(os.path.join(horse12_path,image_name))

def twomask2img(mask1_path,mask2_path,img_path,out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    framelist=os.listdir(mask1_path)
    framelist.sort()
    ground_mask=np.zeros((1080,1920,3),dtype=np.uint8)
    ground_mask[860:,:,:]=[255,255,255]
    ground_mask_pil=Image.fromarray(ground_mask).convert("L")
    for image_name in framelist:
        mask1=Image.open(os.path.join(mask1_path,image_name))
        mask2=Image.open(os.path.join(mask2_path,image_name))
        image=Image.open(os.path.join(img_path,"0000"+image_name.split(".")[1]+".png"))


        transparent_im = Image.new('RGBA', image.size, (0, 0, 0,0))
        transparent_im.paste(image, (0, 0), mask1)
        transparent_im.paste(image, (0, 0), mask2)
        transparent_im.paste(image, (0, 0), ground_mask_pil)
        transparent_im.save(os.path.join(out_path,"0000"+image_name.split(".")[1]+".png"))
        
def twomask2img_no_overlay(mask1_path,mask2_path,img_path,out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    framelist=os.listdir(mask1_path)
    framelist.sort()
    # ground_mask=np.zeros((1080,1920,3),dtype=np.uint8)
    # ground_mask[860:,:,:]=[255,255,255]
    # ground_mask_pil=Image.fromarray(ground_mask).convert("L")
    for image_name in framelist:
        mask1=Image.open(os.path.join(mask1_path,image_name))
        mask2=Image.open(os.path.join(mask2_path,image_name))
        # image=Image.new("RGB",(mask2.width,mask2.height),"white")
        image=Image.open(os.path.join(img_path,"0000"+image_name.split(".")[1]+".png"))


        transparent_im = Image.new('RGBA', image.size, (0, 0, 0,0))
        transparent_im.paste(image, (0, 0), mask1)
        transparent_im.paste(image, (0, 0), mask2)
        # transparent_im.paste(image, (0, 0), ground_mask_pil)
        transparent_im.save(os.path.join(out_path,"0000"+image_name.split(".")[1]+".png"))


def img_move(img_move_path,img_move_out):
    if not os.path.exists(img_move_out):
        os.mkdir(img_move_out)

    image=cv2.imread(img_move_path)#448,896 1920 1080

    window=(567,1018)

    new_frame=[]

    for i in range ((image.shape[1]-window[1])):
        win=i
        new_img=image[:,(image.shape[1]-window[1]-win):(image.shape[1]-win)]
        new_img=cv2.resize(new_img,(1024,576))
        cv2.imwrite(os.path.join(img_move_out,"%08d"%(i)+".png"),new_img)
        new_frame.append(new_img)
        print("over")
    print("over-------")




def pull_mask(rgba_path,alpha_path):
    if not os.path.exists(alpha_path):
        os.mkdir(alpha_path)
    for rgba_img in os.listdir(rgba_path):
        # image=cv2.imread(os.path.join(rgba_path,rgba_img),cv2.IMREAD_UNCHANGED)#448,896
        # alpha=image[:,:,-1]
        # cv2.imwrite(os.path.join(alpha_path,rgba_img),alpha)
        image=Image.open(os.path.join(rgba_path,rgba_img))#448,896
        r,g,b,alpha=image.split()
        alpha.save(os.path.join(alpha_path,rgba_img))

def to_frame(video_path: str):

    res = []
    framelist=os.listdir(video_path)
    framelist.sort()
    for image in framelist:
        image=cv2.imread(os.path.join(video_path,image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res.append(image)

    return res

def rename_file(old_name, new_name):
    os.rename(old_name, new_name)
    
def rename_batch(video_path):
    for image in os.listdir(video_path):
        rename_file(os.path.join(video_path,image),os.path.join(video_path,"%08d"%(int(image.split(".")[0]))+".jpg"))

def modify_mask(mask_input_path,modifyed_mask_out_path):
    if not os.path.exists(modifyed_mask_out_path):
        os.mkdir(modifyed_mask_out_path)
    for mask_input in os.listdir(mask_input_path):
        image=cv2.imread(os.path.join(mask_input_path,mask_input))#448,896
        # alpha=image[:,:,-1]
        # cv2.imwrite(os.path.join(alpha_path,rgba_img),alpha)
        # image=Image.open(os.path.join(mask_input_path,mask_input))#448,896
        image[:640,:,:]=[0,0,0]
        # image[1000:,:,:]=[0,0,0]
        # r,g,b,alpha=image.split()
        # madifted_mask.save(os.path.join(modifyed_mask_out_path,mask_input))
        cv2.imwrite(os.path.join(modifyed_mask_out_path,mask_input),image)

def erode_dilate(erode_dilate_input_path,out_path):
    kernel = np.ones((4, 4), dtype=np.uint8)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for mask_input in os.listdir(erode_dilate_input_path):
        image=cv2.imread(os.path.join(erode_dilate_input_path,mask_input))#448,896
        # alpha=image[:,:,-1]
        # cv2.imwrite(os.path.join(alpha_path,rgba_img),alpha)
        # image=Image.open(os.path.join(mask_input_path,mask_input))#448,896
        # image[:640,:,:]=[0,0,0]
        # image[1000:,:,:]=[0,0,0]
        # r,g,b,alpha=image.split()
        # madifted_mask.save(os.path.join(modifyed_mask_out_path,mask_input)
        image = cv2.dilate(image, kernel, iterations=5)
        cv2.imwrite(os.path.join(out_path,mask_input),image)
    
    # def dilate_mask(mask, kernel_size, iterations=1):
    #     # Create a kernel for dilation
    #     kernel = np.ones((kernel_size, kernel_size), np.uint8)

    #     dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    #     return dilated_mask


    # img = cv2.imread('2.png', 0)  # 0: 读入时转为黑白
    # img = cv2.resize(img, (512, 512))  # 尺寸伸缩
    # thr, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)  # 阈值二值化
    

    # # 定义卷积核mask
    # kernel = np.ones((3, 3), dtype=np.uint8)  # 方形核

    # # 腐蚀 erode == 让黑暗扩散
    # # 如果周围没有黑色，该点就不变
    # # 如果周围存在黑色，该点就变成黑色
    # # params: img, kernel, iterations=1
    # img2 = cv2.erode(img, kernel)
    # img3 = cv2.erode(img, kernel, iterations=3)

    # # 膨胀 dilate == 让光芒扩散
    # # 如果周围存在白色，该点就变成白色
    # # 如果周围没有白色，该点就不变
    # # params: img, kernel, iterations=1
    # img4 = cv2.dilate(img, kernel)
    # img5 = cv2.dilate(img, kernel, iterations=3)

    # # 先腐蚀后膨胀的应用案例：去除白色毛刺
    # img6 = cv2.dilate(cv2.erode(img, kernel), kernel)
    # # 先膨胀后腐蚀的应用案例：去除黑色毛刺
    # img7 = cv2.erode(cv2.dilate(img, kernel), kernel)img = cv2.imread('2.png', 0)  # 0: 读入时转为黑白
    # img = cv2.resize(img, (512, 512))  # 尺寸伸缩
    # thr, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)  # 阈值二值化
    # print(img.max())
    # print(img.min())

    # # 定义卷积核mask
    # kernel = np.ones((3, 3), dtype=np.uint8)  # 方形核

    # # 腐蚀 erode == 让黑暗扩散
    # # 如果周围没有黑色，该点就不变
    # # 如果周围存在黑色，该点就变成黑色
    # # params: img, kernel, iterations=1
    # img2 = cv2.erode(img, kernel)
    # img3 = cv2.erode(img, kernel, iterations=3)

    # # 膨胀 dilate == 让光芒扩散
    # # 如果周围存在白色，该点就变成白色
    # # 如果周围没有白色，该点就不变
    # # params: img, kernel, iterations=1
    # img4 = cv2.dilate(img, kernel)
    # img5 = cv2.dilate(img, kernel, iterations=3)

    # # 先腐蚀后膨胀的应用案例：去除白色毛刺
    # img6 = cv2.dilate(cv2.erode(img, kernel), kernel)
    # # 先膨胀后腐蚀的应用案例：去除黑色毛刺
    # img7 = cv2.erode(cv2.dilate(img, kernel), kernel)


if __name__ == "__main__":

    rgba_path="/home/ubuntu/sws/data/testsce5/scene5all"#"/home/ubuntu/sws/data/aliyun/Scene5horse1"
    alpha_path="/home/ubuntu/sws/data/horse_mask/scene5all-alpha"

    img_move_path="/home/ubuntu/liudong/diffusers-Rerender-rerender/video/sce52inpaint-tail_diffout_bj_8/3.jpg" #"/home/ubuntu/liudong/diffusers-Rerender-rerender/video/sce52inpaint-tail_diffout17/1.jpg"
    img_move_out="/home/ubuntu/liudong/diffusers-Rerender-rerender/video/moved_imgs12"
    
    mask_path="/home/ubuntu/sws/data/horse_mask/horse12mask1"
    add_ground_mask_path="/home/ubuntu/sws/data/horse_mask/horse1mask_ground"
    modifyed_mask_out_path="/home/ubuntu/sws/data/horse_mask/horse1mask_ground_modifyed1"

    img_path="/home/ubuntu/sws/data/aliyun/Scene5horse2"
    overlay_path="/home/ubuntu/sws/data/horse_mask/overlay_horse2_111"
    bg_path="/home/ubuntu/liudong/diffusers-Rerender-rerender/video/moved_imgs12"
    blend_out_path="/home/ubuntu/sws/data/horse_mask/blend_out7"
    horse1_path="/home/ubuntu/sws/data/horse_mask/horse1mask"
    horse2_path="/home/ubuntu/sws/data/horse_mask/horse2mask"
    horse12_path="/home/ubuntu/sws/data/horse_mask/horse2mask"
    two_horse_overlay_out="/home/ubuntu/sws/data/horse_mask/overlay_horse2_111"
    horse2="/home/ubuntu/sws/data/horse_mask/overlay_horse2_111"
    horse1="/home/ubuntu/sws/data/horse_mask/overlay_horse1_111"
    two_horse_mask_out="/home/ubuntu/sws/data/horse_mask/two_horse_mask2"

    output_frames="/home/ubuntu/liudong/diffusers-Rerender-rerender/video/sce52inpaint-tail_diffout_basemodel_41"
    output_path="/home/ubuntu/liudong/diffusers-Rerender-rerender/video/sce52inpaint-tail_diffout21.mp4"

    erode_dilate_input_path="/home/ubuntu/sws/data/horse_mask/horse1mask_ground"
    erode_dilate_out_path="/home/ubuntu/sws/data/horse_mask/horse1mask_ground_dilate1"

    # pull_mask(rgba_path,alpha_path)
    # img_move(img_move_path,img_move_out)

    # add_ground_mask(horse1_path,add_ground_mask_path)

    # mask2img(horse12_path,img_path,overlay_path)

    # horse12_mask_add(horse1_path,horse2_path,horse12_path)
    # twomask2img_no_overlay(horse1_path,horse2_path,img_path,two_horse_mask_out)

    # blend_img_1(bg_path,horse1,blend_out_path)
    # blend_img(bg_path,horse1,horse2,blend_out_path)
    # blend_img(bg_path,two_horse_overlay_out,blend_out_path)
    #blend_img(bg_path,overlay_path,blend_out_path)

    # rename_batch(output_frames)
    # export_to_video(
    # to_frame(output_frames),output_path, 10)
    # modify_mask(add_ground_mask_path,modifyed_mask_out_path)
    erode_dilate(erode_dilate_input_path,erode_dilate_out_path)