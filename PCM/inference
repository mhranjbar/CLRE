import numpy as np
from tensorflow.keras.models import load_model
import cv2


inputSize = 224
model = load_model('MobilNet-pcm_model.h5')

def inference(filepath):

    gray = cv2.imread(filepath)
    gray = cv2.resize(gray, (inputSize,inputSize))
    gray = gray/255
    gray = np.array(gray)
    gray = np.reshape(gray, (1,inputSize,inputSize,3))

    predictions = model.predict(gray)
    predictions = (predictions * 29) + 13


    for a, b in enumerate(predictions[0]):
        T = "T" + str(a+1)
        print(T,": ","%.2f"%b," C")


filepath = "dataset/" + '562.jpg'
inference(filepath)
