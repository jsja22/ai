import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
from numpy import asarray
from PIL import Image

train_img=[]
for i in range(50000):
    filepath='C:/data/computer_vision2/dirty_mnist_2nd/%05d.png'%i
    #image=Image.open(filepath)

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE
    image_resize = cv2.resize(image, (64, 64)) # 이미지 크기를 50x50 픽셀로 변경
    #img = cv2.imshow('original', image)
    #cv2.waitKey(0)

    #254보다 작고 0이아니면 0으로 만들어주기
    image2 = np.where((image_resize <= 254) & (image_resize != 0), 0, image_resize)
    #cv2.imshow('filterd', image2)

    image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    #cv2.imshow('dilate', image3)
    #dilate -> 이미지 팽창
    image4 = cv2.medianBlur(src=image2, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
    #cv2.imshow('median', image4)
    #medianBlur->커널 내의 필터중 밝기를 줄세워서 중간에 있는 값으로 현재 픽셀 값을 대체

    image5 = image4 - image2
    #cv2.imshow('sub', image5)

    image_data=asarray(image5)
    train_img.append(image_data)
    print("complite : {}".format(i))
np.save('C:/data/computer_vision2/npy/train_data2.npy', arr=train_img)


test_img=[]
for i in range(50000,55000):
    filepath='C:/data/computer_vision2/test_dirty_mnist_2nd/%05d.png'%i
    #image=Image.open(filepath)
    
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE
    image_resize = cv2.resize(image, (64, 64)) # 이미지 크기를 50x50 픽셀로 변경
    #img = cv2.imshow('original', image)
    #cv2.waitKey(0)

    #254보다 작고 0이아니면 0으로 만들어주기
    image2 = np.where((image_resize <= 254) & (image_resize != 0), 0, image_resize)
    #cv2.imshow('filterd', image2)

    image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    #cv2.imshow('dilate', image3)
    #dilate -> 이미지 팽창
    image4 = cv2.medianBlur(src=image2, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
    #cv2.imshow('median', image4)
    #medianBlur->커널 내의 필터중 밝기를 줄세워서 중간에 있는 값으로 현재 픽셀 값을 대체

    image5 = image4 - image2
    #cv2.imshow('sub', image5)

    image_data=asarray(image5)
    test_img.append(image_data)
    print("complite : {}".format(i))

np.save('C:/data/computer_vision2/npy/test_data2.npy', arr=test_img)

