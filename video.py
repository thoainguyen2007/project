#frame from file video
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
def read_testing_data():
    features_dict = {}
    pretrained = PretrainModel()
    pil_image = Image.fromarray(image).convert("RGB")
    features_dict = pretrained.get_feature(pil_image)
    return features_dict



import cv2
vidcap = cv2.VideoCapture('test.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  X=read_testing_data()
  y_test=model_load.predict(X.reshape(1,-1))#(1,-1) tức là (1,576)
  if y_test[0]==1:
    pygame.mixer.music.play()
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1



# import pygame
# # for i in range(2):
# #   pygame.init()
# #   warning=pygame.mixer.Sound("warning.wav")
# #   warning.play()
# pygame.mixer.init()
# pygame.mixer.music.load("warning.mp3")
# pygame.mixer.music.play()
# while True:
# 	if input() == 'e':
# 		pygame.mixer.music.stop()
# 		break


