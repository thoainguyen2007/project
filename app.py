import os
import torch
import torchvision.transforms as transforms
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import PretrainModel
from model import read_training_data_label 
from model import read_training_data 

#Get X,y
X=read_training_data()
X=list(X.values())
X=np.array(X)
y=read_training_data_label()
y=list(y.values())
y=np.array(y)

#Train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)

#Save model
import pickle
filename = 'model.pickle'
pickle.dump(model, open(filename, "wb"))