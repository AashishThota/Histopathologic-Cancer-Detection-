from sklearn.utils import shuffle

import pandas as pd

from PIL import Image

import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from sklearn.preprocessing import LabelEncoder

from keras.constraints import maxnorm

from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

import matplotlib.pyplot as plt

import keras

data=pd.read_csv('train_labels.csv')

# take a random sample of class 0 with size equal to num samples in class 1

df_0 = data[data['label'] == 0].sample(5000, random_state = 101)

# filter out class 1

df_1 = data[data['label'] == 1].sample(5000, random_state = 101)



# concat the dataframes

df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)

# shuffle

data= shuffle(df_data)



X=[]

for i in range(0,len(data)):

    img=data.iloc[i]['id']

    img=str(img)

    impath=img+'.tif'

    im = Image.open(impath)

    X.append(np.asarray(im, dtype=np.uint8))

X=np.asarray(X,dtype=np.float32)



X = X / 255.0

Y=data['label'];



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)



Y_train = np_utils.to_categorical(Y_train)

Y_test=np_utils.to_categorical(Y_test)

num_classes=Y_test.shape[1]

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(96, 96,3), activation='relu',padding='same',kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), input_shape=(96,96,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), input_shape=(96,96,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dropout(0.2))



model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Dense(num_classes,activation='softmax'))

# Compile model

epochs = 15

lrate = 0.01

decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=epochs, batch_size=300)

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'],loc='upper left')

plt.show()

scores = model.evaluate(X_test, Y_test, verbose=0)

print("the test image is")

plt.imshow(X_test[0,:,:],cmap='gray')

plt.show()

predict= model.predict_classes(X_test[[0],:])



print(predict)

print("Accuracy: %.2f%%" % (scores[1]*100))