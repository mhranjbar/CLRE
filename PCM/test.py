import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt


inputSize = 224

model = load_model('MobilNet-pcm_model.h5')

################ Get data ###################

f = open("output.txt","w")

x_train = []
y_train = []

i = 0

for fileNumber in range(34,891): 
    filepath = "dataset/" + str(fileNumber) + ".jpg"
    print(filepath)
    #print(landmarks[i])

    gray = cv2.imread(filepath)
    #gray = cv2.GaussianBlur(gray,(3,3),0)
    #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (inputSize,inputSize))
    gray = gray/255

    gray = np.array(gray)
    #gray = np.reshape(gray, (3,inputSize,inputSize,1))
    gray = np.reshape(gray, (1,inputSize,inputSize,3))
    #gray = np.expand_dims(gray, -1)


    predictions = model.predict(gray)
    predictions = (predictions * 29) + 13


    T = []
    for a, b in enumerate(predictions[0]):

        if a == 0:
            T = str(b)
        else:
            T = T + "," + str(b)

    line = str(fileNumber) + "," + T + "\n"
    f.write(line)



    i +=1

f.close()