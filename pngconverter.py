'''
Convert an image Dataset from one extension to another.
Author: Alfonso Navarro
Date: 23/12/2022
Version: 1.0
'''
from PIL import Image
import os

PATH = 'C:/Users/anavarro4/Downloads/WhatsApp Unknown 2023-01-06 at 10.47.08 AM'
NEW_PATH = 'C:/Users/anavarro4/Downloads/WhatsAppPNG'
File_extension = '.png'
folders = os.listdir(PATH)
files = []

for folder in folders:
    i = 1
    for file in os.listdir(PATH+"/"+folder):
        im = Image.open(PATH+"/"+folder+"/"+file)
        im.save(NEW_PATH+"/"+folder+"/"+'image'+str(i)+File_extension)
        i+=1

