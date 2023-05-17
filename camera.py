#frame from camera
import cv2
import os
import torch
import torchvision.transforms as transforms
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import PretrainModel
import numpy as np
import pygame
model_load = pickle.load(open('model.pickle', "rb"))
webCam=cv2.VideoCapture(0)
currentframe=0

def read_testing_data():
    features_dict = {}
    pretrained = PretrainModel()
    pil_image = Image.fromarray(frame).convert("RGB")
    features_dict = pretrained.get_feature(pil_image)
    return features_dict
# success, framebackground=webCam.read()
# cv2.imwrite('hinh.jpg',framebackground)
pygame.mixer.init()
pygame.mixer.music.load("warning.mp3")
next=60
while True:
    success, frame=webCam.read()#frame là biến chứa ảnh, dùng biến này để predict
    
    X=read_testing_data()
    y_test=model_load.predict(X.reshape(1,-1))#(1,-1) tức là (1,576)
    # cv2.imwrite('hinh'+str(currentframe)+'.jpg',frame)
    if y_test==1:
        print('1')
        if next>=60:    #tức là đã loop hơn 60 lần từ lần cảnh báo gần nhất, điều này đảm báo đã cảnh báo xong, có thể cảnh báo lại
            
            pygame.mixer.music.play()
            next=0      #biến next cho biết số lần loop sau lần cảnh báo gần nhất
    next+=1

    
    cv2.imshow('output',frame)   #hiện camera (ko nhất thiết phải có,không ảnh hưởng đến code)

    currentframe+=1
    if cv2.waitKey(1)& 0xFF==ord('q'):#bấm q trên bàn phím để ngừng
        break
webCam.release()
cv2.destroyAllWindows()

