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
    filename_violet = folder + "/Masks/" + image_name + 'violet' + '.tif'
    filename_green = folder + "/Masks/" + image_name + 'green' + '.tif'
    filename_red = folder + "/Masks/" + image_name + 'red' + '.tif'

    print(filename_green)

    cv2.imwrite(filename_violet, labels_violet)
    cv2.imwrite(filename_green, labels_green)
    cv2.imwrite(filename_red, labels_red)
