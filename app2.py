import cv2
import pickle
from PIL import Image
from model import PretrainModel
import pygame

model_load = pickle.load(open('model.pickle', "rb"))
webCam=cv2.VideoCapture(0)
def read_testing_data():  #lấy ra shape thỏa model
    pretrained = PretrainModel()
    pil_image = Image.fromarray(frame).convert("RGB")
    X= pretrained.get_feature(pil_image).reshape(1,576)
    return X
pygame.mixer.init()
pygame.mixer.music.load("warning.mp3")
count=0
ROI = ((100, 20), (540, 460))   #lấy 2 bên vào 100 pixel, trên dưới là 20 pixel
while True:
    success, frame=webCam.read()#frame là biến chứa ảnh, dùng biến này để predict
    frame = frame[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0], :]
    X=read_testing_data()  
    y_test=model_load.predict(X)
    if y_test==1:
        count+=1
    if count>=60:   
        pygame.mixer.music.play()
        count=0
    cv2.imshow('output',frame)   
    if cv2.waitKey(1)& 0xFF==ord('q'):#bấm q trên bàn phím để ngừng
        break
webCam.release()
cv2.destroyAllWindows()