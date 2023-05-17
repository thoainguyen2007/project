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
warn=[]
pygame.mixer.init()
pygame.mixer.music.load("warning.mp3")
while True:
    success, frame=webCam.read()#frame là biến chứa ảnh, dùng biến này để predict
    
    X=read_testing_data()
    y_test=model_load.predict(X.reshape(1,-1))#(1,-1) tức là (1,576)
    # cv2.imwrite('hinh'+str(currentframe)+'.jpg',frame)
    currentframe+=1
    if currentframe==100:
        pygame.mixer.music.play()
    cv2.imshow('output',frame)   #hiện camera(ko nhất thiết phải có,không ảnh hưởng đến code)


    if cv2.waitKey(1)& 0xFF==ord('q'):#bấm q trên bàn phím để ngừng
        break
webCam.release()
cv2.destroyAllWindows()

