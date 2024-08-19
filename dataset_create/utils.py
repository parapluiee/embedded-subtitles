import os, shutil
from PIL import Image, ImageFilter
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def pillow_edge(img):
    img = img.convert("L")
    return img.filter(ImageFilter.FIND_EDGES)

def edge_detect(img):
    return pillow_edge(img)
