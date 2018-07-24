import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

images = []
measurements = []
correction = 0.4

def read_image(source_path, measurement):
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # add image and corresponding measurement to dataset
    images.append(image)
    measurements.append(measurement)
    # flip the image and add it to dataset with appropriate measurement
    images.append(cv2.flip(image, 1))
    measurements.append(-measurement)

with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip the header line
    for line in reader:
        read_image(line[0], float(line[3]))              # center camera image
        read_image(line[1], float(line[3]) + correction) # left camera image
        read_image(line[2], float(line[3]) - correction) # right camera image

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
# Preprocess 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# Nvidia CNN Architecture
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Configure and Train the model
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save('model.h5')
exit()