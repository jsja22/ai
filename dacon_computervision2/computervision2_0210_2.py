import numpy as np
import PIL
from numpy import asarray
from PIL import Image


#dirty_mnist_2nd ->5만개의 훈련 데이터
#dirty_mnist_2nd_answer -> 5만장의 사진마다 각각의 y값 존재하는 알파벳 1 없는 알파벳 0

#test_dirty_mnist_2nd-> 5000
#submission ->(50000~55000)까지 y값을 찾는것이 목표!

img=[]
for i in range(50000):
    filepath='C:/data/computer_vision2/dirty_mnist_2nd/%05d.png'%i
    image=Image.open(filepath)
    image_data=asarray(image)
    img.append(image_data)


np.save('C:/data/computer_vision2/npy/train_data.npy', arr=img)