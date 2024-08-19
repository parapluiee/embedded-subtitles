from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pathlib import Path
import random
import string
import utils
import pickle
working_dir = 'data/'
width = 640
height = 360
box_buffer = 5
def add_text(image, file_name, position, text, color, output_dir, font, box_coords, box_show=False, img_show=False):
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    draw.text(position, text, color, anchor='md', font=font, align='center')
    if (box_show==True):
        draw.rectangle(box_coords)
    if (img_show):
        img.show()
    img = utils.edge_detect(img)
    img.save(output_dir + '/' + file_name)
    img.close()
    

#generative functionss
def rand_position(wid, hei):
    #prob need to tune this to be realistic, i.e not 0 or 720
    y = random.randint(50, hei-50)
    return (wid/2, y)

def rand_text(pixels):
    max_length = int(width / pixels)
    lines = random.randint(1, 4)
    string = ""
    max_line_length = 0
    for i in range(lines):
        line_length = random.randint(1, max_length)
        for j in range(line_length):
            string+=rand_char()
        if (i != lines-1): 
            string+='\n'
    return string
def rand_char():
    #create own proababilities based on natural text, i.e add spaces and punctuation
    return random.choice(string.ascii_letters)

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
    return random.randint(10, 30)
def rand_font(font_size):
    return ImageFont.truetype("arial.ttf", font_size)

def det_coords(text, font, position):
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
    box = [int(position[0] - int(length/2) - box_buffer), int(position[1] - (metrics[0] + metrics[1]) * lines - box_buffer), int((position[0] + length/2) + box_buffer), int(position[1] + box_buffer)]
    return box

def run_in_dir(input_dir, output_dir, max_imgs=0):
    data_dict = dict()
    utils.clear_directory(output_dir)
    input_path = Path(input_dir)
    print("Start iteration\n")
    id_num = 0
    for image in input_path.iterdir():
        if id_num % 1000 == 0:
            print("Text insertion imgage: " + str(id_num))
        if max_imgs != 0 and id_num == max_imgs:
            return data_dict
        for j in range(5):
            pos = rand_position(width, height)
            font_size = rand_f_size()
            font = rand_font(font_size)
            text = rand_text(font_size)
            box_coords = det_coords(text, font, pos)
            color = rand_color("white")
            t = False
            file_name = image.stem + '_' + str(j) + '_' + str(id_num) + image.suffix
            add_text(image, file_name, pos, text, color, output_dir, font, box_coords, t)
            data_dict[file_name] = box_coords
        id_num+=1
    return data_dict

def dict_pickle(data_dict, name):
    dict_file = open(working_dir + name + '.obj', 'wb')
    pickle.dump(data_dict, dict_file)
    dict_file.close

def main():
    notxt_dir = 'jpgs'
    train_dir = 'w_text_train'
    valid_dir = 'w_text_valid'

    data_dict = dict()
    train_dict = run_in_dir(working_dir + notxt_dir, working_dir + train_dir, 2000)
    valid_dict = run_in_dir(working_dir + notxt_dir, working_dir + valid_dir, 500)
     
    dict_pickle(train_dict, 'train_data')
    dict_pickle(valid_dict, 'valid_data')

def get_metrics(font_size):
    font = font=ImageFont.truetype("arial.ttf", font_size)
    return font.getmetrics()
font = rand_font(rand_f_size())
main()
