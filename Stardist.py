from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk
import tkinter.filedialog as fd
from stardist.models import StarDist2D

# prints a list of available models
# StarDist2D.from_pretrained()

paths = ["C:/Users/Modern/Desktop/Python/Stardist/Images/ApNec_300-20_3.tif"]
# root = tk.Tk()
# path = fd.askopenfilenames(parent=root, title='Open images')
# print(path)
# paths = ["C:/Users/Modern/Desktop/Python/Stardist/Images/ApNec_300-20_3.tif",
# "C:/Users/Modern/Desktop/Python/Stardist/Images/ApNec_600-20_3.tif",
# "C:/Users/Modern/Desktop/Python/Stardist/Images/ApNec_600-20_1.tif" ]

model = StarDist2D.from_pretrained('2D_versatile_fluo')
for p in paths:
    mylist = []
    loaded,images = cv2.imreadmulti(mats = mylist, filename = p, flags = cv2.IMREAD_ANYCOLOR )
    image_violet = images[0]
    image_green = images[3]
    image_red = images[4]

    labels_violet, _ = model.predict_instances(normalize(image_violet[:,:,2]))
    labels_green, _ = model.predict_instances(normalize(image_green[:,:,2]))
    labels_red, _ = model.predict_instances(normalize(image_red[:,:,2]))   

# cv2.imshow('image',labels_violet)
# cv2.imshow('image',labels_red)
# cv2.imshow('image',labels_green)
# cv2.waitKey(0)

plt.subplot(1,2,2)
# plt.imshow(render_label(labels, img=img))
plt.imshow(labels_green)
plt.axis("off")
plt.title("prediction + input overlay")
plt.show()


# image = images[3]
# img = image[:,:,2]
# print(np.shape(img))

# cv2.imshow("mylist[1]",img[:,:,2])
# cv2.waitKey()

# creates a pretrained model


# labels, _ = model.predict_instances(normalize(img)) 

# # print(np.shape(labels))

# plt.subplot(1,2,1)
# plt.imshow(img, cmap="gray")
# plt.axis("off")
# plt.title("input image")

# plt.subplot(1,2,2)
# # plt.imshow(render_label(labels, img=img))
# plt.imshow(labels)
# plt.axis("off")
# plt.title("prediction + input overlay")

# plt.show()