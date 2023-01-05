from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk
import tkinter.filedialog as fd
from stardist.models import StarDist2D
import os

def file_folder_name(path):
    imagename_without_ext = os.path.splitext(os.path.basename(path))[0]
    dirname = os.path.dirname(path)

    return dirname, imagename_without_ext


# prints a list of available models
# StarDist2D.from_pretrained()

# paths = ["C:/Users/Modern/Desktop/Python/Stardist/Images/ApNec_300-20_3.tif"]
root = tk.Tk()
paths = fd.askopenfilenames(parent=root, title='Open images')

# paths = ["C:/Users/Modern/Desktop/Python/Stardist/Images/ApNec_300-20_3.tif",
# "C:/Users/Modern/Desktop/Python/Stardist/Images/ApNec 600-20 3.tif",
# "C:/Users/Modern/Desktop/Python/Stardist/Images/ApNec 600-20 1.tif" ]
n = 0
model = StarDist2D.from_pretrained('2D_versatile_fluo')
for p in paths:
    mylist = []
    loaded,images = cv2.imreadmulti(mats = mylist, filename = p, flags = cv2.IMREAD_ANYCOLOR )
    image_violet, image_green, image_red = images[0],images[3],images[4]

    labels_violet, _ = model.predict_instances(normalize(image_violet[:,:,2]))
    labels_green, _ = model.predict_instances(normalize(image_green[:,:,2]))
    labels_red, _ = model.predict_instances(normalize(image_red[:,:,2]))   

    name_suffixes = ['violet','green','red']

    folder,image_name = file_folder_name(p)
    filename_violet_mask = folder + "/Masks/" + str(n) + image_name + 'violet_mask' + '.tif'
    filename_green_mask = folder + "/Masks/" + str(n) + image_name + 'green_mask' + '.tif'
    filename_red_mask = folder + "/Masks/" + str(n) + image_name + 'red_mask' + '.tif'

    filename_violet = folder + "/Masks/" + str(n) + image_name + 'violet_orig' + '.tif'


    print(filename_green_mask)

    cv2.imwrite(filename_violet_mask, labels_violet)
    cv2.imwrite(filename_green_mask, labels_green)
    cv2.imwrite(filename_red_mask, labels_red)

    cv2.imwrite(filename_violet, image_violet[:,:,2])

    n = n+1
