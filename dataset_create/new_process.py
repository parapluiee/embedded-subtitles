import cv2
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import random
import re
import os, shutil
import string
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import torch
from object_local import main
from object_local import flood_fill_square
def pillow_edge(imgs):
    out = []
    for img in imgs:
        img = img.convert("L")
        out.append(img.filter(ImageFilter.FIND_EDGES))
    return out


def add_text(image, position, text, color, font, box_coords, box_show=False, img_show=False):
    draw = ImageDraw.Draw(image)
    draw.text(position, text, color, anchor='md', font=font, align='center')
    if (box_show==True):
        draw.rectangle(box_coords)
    if (img_show):
        image.show()
    return image
    
#returns center of image, with random height
def rand_position(wid, hei):
    #prob need to tune this to be realistic, i.e not 0 or 720
    #higher num = lower, 
    y = random.randint(hei - int(hei/5), hei)
    return (wid/2, y)
char_set = string.ascii_uppercase + string.digits + string.ascii_lowercase + " "
def rand_text(font_size, width, lines):
    max_length = int(width / (font_size /1.5))
    string = ""
    max_line_length = 0
    for i in range(lines):
        line_length = random.randint(1, max_length)
        if (max_line_length < line_length):
            max_line_length = line_length
        string += ''.join(random.choices(char_set, k=line_length))
        if (i != lines-1): 
            string+='\n'
    return (string, max_line_length)

def rand_color(choice):
    if choice == "white":
        return (255,255,255)
    elif choice == "full":
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
    elif choice == "w/b":
        if (random.randin(0, 1)):
            return (255,255,255)  
        else: 
            return (0, 0, 0)

def rand_f_size():
    return random.randint(10, 35)
def rand_font(font_size):
    return ImageFont.truetype("arial.ttf", font_size)

#x1, y1, x2, y2
def det_coords(text, font, position, box_buffer=0):
    split = text.split('\n')
    l = 0
    longest_line = ''
    lines = len(split)
    for line in split:
        if len(line) > l:
            l = len(line)
            longest_line = line
    metrics = font.getmetrics()
    length = font.getlength(longest_line) 
    box = [int(position[0] - int(length/2) - box_buffer), int(position[1] - (metrics[0] + metrics[1]) * lines - box_buffer), int((position[0] + length/2) + box_buffer), int(min(position[1], position[1] + 2 * box_buffer))]
    return box

def add_training_data(images):
    coords = list()
    out = []
    width, height = images[0].size
    for image in images:
        lines = random.randint(1, 3)
        font_size = rand_f_size()
        font = rand_font(font_size)
        text, max_length = rand_text(font_size, width, lines)
        pos = rand_position(width, height)
        box_coords = det_coords(text, font, pos)
        color = rand_color("white")
        t = False
        out.append(add_text(image, pos, text, color, font, box_coords, box_show=t))
        coords.append(box_coords)
    return out, coords

def cv2_to_pil(images):
    output = []
    for image in images:
        output.append(Image.fromarray(image))
    return output

#batch_size: if True, does all images
#            if an integer, returns that number of images
def save_images(cap, batch_size=True, FRAMES_PER_IMAGE=10):
    images = []
    frame_num=-1
    while(batch_size == True or len(images) < batch_size):
        frame_num += 1
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            cv2.destroyAllWindows()
            return images
        if (frame_num % FRAMES_PER_IMAGE != 0):
            continue
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return images

def reduce_imgs(imgs, red_height=128):
    width, height = imgs[0].size
    ratio = width/height
    size = int(red_height * ratio), red_height
    out = []
    for img in imgs:
        img.thumbnail(size)
        #creates just dots, kinda sick
        out.append(img.convert('1'))
    return out

def edge_det_compr(imgs, reduce=True):
    edged = pillow_edge(imgs)
    if (reduce):
        edged = reduce_imgs(edged)
    return edged


def create_image_data(cap, batch_size=True, train=False, mask=False):
    images = save_images(cap, batch_size=batch_size)
    if (len(images) != 0):
        images = cv2_to_pil(images)
        if (train == True):
            images, coords = add_training_data(images) 
            old_size = images[0].size
        else:
            clean_images = images
        images = edge_det_compr(images, reduce=False)
        new_size = images[0].size
        if (mask):
            ratio = new_size[1] / old_size[1]
            coords = coords_to_mask(coords, ratio, new_size)
        if (train == True):
            images = zip(images, coords)
        else:
            images = zip(images, clean_images)
        return images

def run_on_dev(model, video_path):
    model.eval()
    cap = cv2.VideoCapture(video_path)
    #maybe necessary to have hyperparam frame_split (i.e 10)
    batch=0
    while (cap.isOpened()):
        images, clean_images = zip(*list(create_image_data(cap, batch_size=32)))
        images_tensor = torch.unsqueeze(torch.tensor(np.array(images)).float().squeeze(), 1)
        masks = model(images_tensor)

        masks = (masks.detach().numpy() > .5)
        #print("clean_width: ", clean_width)
        #print("mask_width: ", height)
        for i in range(len(images)):
            path = "batch_" + str(batch) + "_" + str(i).zfill(4) + ".jpg"
            clean_images[i].save("data/predict/clean_images/" + path) 


            masks[i][0] = flood_fill_square(masks[i][0], buffer = 10)
            swapped_mask = np.swapaxes(np.swapaxes(np.array(masks[i]),0,2), 0, 1).squeeze()
            #swapped_mask = np.swapaxes(np.swapaxes(np.array(masks[i]),0,2), 0, 1)
            Image.fromarray(edge_det_compr([clean_images[i]], reduce=False)[0] * swapped_mask).save("data/predict/cropped/" + path)
        return False
        batch+=1
        




def coords_to_mask(coords, ratio, shape):
    mask = np.zeros((len(coords), shape[1], shape[0]))
    imgs, hei, wid = mask.shape
    #print(coords)
    for i in range(imgs):
        tl, tt, br, bb = coords[i]
        for j in range(int(tl * ratio), int(br * ratio)):
            for k in range(int(tt * ratio), int(bb * ratio)):
                mask[i][k][j] = True
    return mask
    
    return coords
def create_train_data(video_path):
    cap = cv2.VideoCapture(video_path)
    #zip of image and original coordinates
    image_coords = create_image_data(cap, train=True, mask=True)
    return image_coords

def training_data_dir(dir_path):
    videos_dir = Path(dir_path)
    img_arrays = list()
    coords_arrays = list()
    for video_path in videos_dir.iterdir():
        #possible coords is a mask
        image_coords = create_train_data(video_path)
        images, coords = zip(*list(image_coords))
        img_arrays += images
        coords_arrays += coords
        #Image.fromarray(coords[0]).show()
        #images[0].show()
    numpy_imgs = np.array(img_arrays)
    numpy_coords = np.array(coords_arrays)
    #print(numpy_imgs.shape)
    np.save("data/input_shape.npy", numpy_imgs.shape)
    np.save("data/label_shape.npy", numpy_coords.shape)
    np.save("data/input.npy", numpy_imgs.reshape(numpy_imgs.shape[0], -1))
    np.save("data/labels.npy", numpy_coords.reshape(numpy_coords.shape[0], -1))


#train model
#training_data_dir("data/videos/")
#model = main()
#torch.save(model, "model.pt")

#predict
#model = torch.load("model.pt")
#vid_path = "data/predict/embedded.mp4"
#run_on_dev(model, vid_path)


def train_on(training_data):
    training_data_dir(training_data)
    model = main()
    torch.save(model, "model.pt")

def pred(video_path):
    model = torch.load("model.pt")
    vid_path = "data/predict/embedded.mp4"
    run_on_dev(model, vid_path)
