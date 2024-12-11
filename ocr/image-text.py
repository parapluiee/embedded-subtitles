from PIL import Image, ImageOps, ImageFilter
from pytesseract import pytesseract
from pathlib import Path

images_dir = Path("../dataset_create/data/predict/cropped")
path_to_tesseract = r"/usr/bin/tesseract"
i=0
for image_path in sorted(images_dir.iterdir()):
    img = Image.open(image_path)
    #img.show() 
    pytesseract.tesseract_cmd = path_to_tesseract
    inv_img = ImageOps.invert(img.filter(ImageFilter.SMOOTH_MORE))
    #inv_img = ImageOps.invert(Image.eval(img, lambda x:0 if x < 200 else x).filter(ImageFilter.SMOOTH_MORE))
 
    if (i % 15 == 0):
        inv_img.show()
    text = pytesseract.image_to_string(inv_img)
    i+=1
    print(text[:-1])
