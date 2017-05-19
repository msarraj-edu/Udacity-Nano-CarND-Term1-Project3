import csv
import cv2
import numpy as np

lines = []
with open('driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = [0.0, 0.2, -0.2]
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = '.\\IMG\\' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])+correction[i]
        measurements.append(measurement)

        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)


X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers.convolutional import MaxPooling2D, Convolution2D


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model_augment.h5')






























