from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pathlib import Path
import random
import string
import utils
import pickle
working_dir = 'add-text/'
notxt_dir = 'no_text_tiny'
wtxt_dir = 'w_text'
width = 640
height = 360
box_buffer = 5
data_dict = dict()
def add_text(image, position, text, color, output_dir, font, text_num, id_num, box_coords, img_show=False):
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    draw.text(position, text, color, anchor='md', font=font)
    draw.rectangle(box_coords)
    if (img_show):
        img.show()
    file_name = image.stem + '_' + str(text_num) + '_' + str(id_num) + image.suffix
    img.save(output_dir + '/' + file_name)
    data_dict[file_name] = box_coords
    

#generative functionss
def rand_position(wid, hei):
    #prob need to tune this to be realistic, i.e not 0 or 720
    y = random.randint(0, hei)
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
    box = [position[0] - int(length/2) - box_buffer, position[1] - (metrics[0] + metrics[1]) * lines - box_buffer, position[0] + int(length/2) + box_buffer, position[1] + box_buffer]
    return box

def run_in_dir(input_dir, output_dir):
    utils.clear_directory(output_dir)
    input_path = Path(input_dir)
    print("Start iteration\n")
    id_num = 1
    for j in range(10):
        pos = rand_position(width, height)
        font_size = rand_f_size()
        font = rand_font(font_size)
        text = rand_text(font_size)
        box_coords = det_coords(text, font, pos)
        color = rand_color("white")
        t = False
        i = 0
        for image in input_path.iterdir():
            if i % 1000 == 0:
                print(i)
            add_text(image, pos, text, color, output_dir, font, j, i, box_coords, t)
            i+=1

def dict_pickle():
    dict_file = open(working_dir + 'img_data.pkl', 'wb')
    pickle.dump(data_dict, dict_file)
    dict_file.close
def main():
    run_in_dir(working_dir + notxt_dir, working_dir + wtxt_dir)
    dict_pickle()

def get_metrics(font_size):
    font = font=ImageFont.truetype("arial.ttf", font_size)
    return font.getmetrics()
font = rand_font(rand_f_size())
main()
