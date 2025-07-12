import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt


inputSize = 224

################ Get data ###################
landmarks_frame = pd.read_csv("dataset/temp.csv") 
#landmarks_frame = pd.read_csv("dataset/tempRaw - Copy.csv")
landmarks = landmarks_frame.iloc[ : , 1:]
landmarks = landmarks.values

#print(landmarks)
x_train = []
y_train = []

i = 0
'''
pathTemp = []
for filepath in glob.iglob('dataset/*.jpg'):
    pathTemp.append(filepath)

pathTemp = sorted(pathTemp)
print("pathTemp: ",pathTemp)
'''

for fileNumber in range(34,891): 
    filepath = "dataset/" + str(fileNumber) + ".jpg"
    #print(filepath)
    #print(landmarks[i])

    gray = cv2.imread(filepath)
    #gray = cv2.GaussianBlur(gray,(3,3),0)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (inputSize,inputSize))
    gray = gray /255

    x_train.append(gray)
    y_train.append(landmarks[i]) # 



    i +=1


x_train = np.array(x_train)
y_train = np.asarray(y_train) #, dtype='uint8')

x_train = np.expand_dims(x_train, -1)


print(x_train.shape)
print(y_train.shape)
print("####################################")

num_classes = 4
input_shape = (inputSize, inputSize, 1)
#############################################

model_save = ModelCheckpoint('best_model.h5',save_best_only=True)


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),      

        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),

        #layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.BatchNormalization(),

        layers.Dense(num_classes)
    ]
)

model.summary()



batch_size = 16
epochs = 220

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])  # "categorical_crossentropy"

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks = [model_save], shuffle = True)

model.save('pcm_model.h5')  # creates a HDF5 file 'pcm_model.h5'


# Plot train vs test accuracy per epoch
plt.figure()
# Use the history metrics
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# Make it pretty
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()



'''
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
'''