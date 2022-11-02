#image size
import cv2
import glob
import os
import numpy as np

datapath = '/home/nallurn1/VQGAN/resize/'
counter = 0

for file in glob.glob('/home/nallurn1/VQGAN/images/*.jpg'):
	im = cv2.imread(file) 
	print(type(im))
	print(im.shape)
	
	#Resizing to 32x32 
	image = np.resize(im, (1, 32, 32, 3))
	cv2.imwrite((str(datapath) + '{}.jpg'.format(counter)), image)
	counter += 1
	cv2.waitKey(0)