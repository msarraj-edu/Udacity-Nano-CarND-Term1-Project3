


import csv
import cv2
import numpy as np
import pandas as pd
def get_image_steering_angle_raw(line):
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = '.\\IMG\\' + filename
    image = cv2.imread(current_path)
    measurement = float(line[3])

    yield image,measurement

def preprocess_augment_data(raw_images, raw_angles):
    yield raw_images,raw_angles

def get_validation_data_generator(data_df, batch_size):
    yield get_training_data_generator(data_df, batch_size)

def get_training_data_generator(data_df, batch_size):
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_angles = np.zeros(batch_size)

    while True:
        for i in range(batch_size):
            '''Shuffle and randomly select any data frame'''
            new_df = data_df.sample(frac=1).reset_index(drop=True)
            idx = np.random.randint(len(new_df))

            data_row = new_df.iloc[[idx]].reset_index()
            '''Get raw image and steering angle'''
            raw_images, raw_angles = get_image_steering_angle_raw(data_row)
            '''Apply augmentation on the image'''
            img, angle = preprocess_augment_data(raw_images, raw_angles)

            batch_images[i] = img
            batch_angles[i] = angle

        yield batch_images, batch_angles

lines = []
# with open('driving_log.csv') as csv_file:
#     reader = csv.reader(csv_file)
#     for line in reader:
#         lines.append(line)
#
# images = []
# measurements = []
#
# for line in lines[:20]:
#     source_path = line[0]
#     filename = source_path.split('\\')[-1]
#     current_path = '.\\IMG\\' + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#     measurement = float(line[3])
#     measurements.append(measurement)
#
# X_train = np.array(images)
# y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64


df = pd.read_csv('driving_log.csv')
msk = np.random.rand(len(df)) < 0.8
training_data = train = df[msk]
validation_data = df[~msk]
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

# training_data,validation_data, training_labels, validation_labels = train_test_split(X_train,y_train, test_size=0.2, random_state= 0)

training_generator = get_training_data_generator(training_data, batch_size=BATCH_SIZE)
validation_generator = get_validation_data_generator(validation_data, batch_size=BATCH_SIZE)

samples_per_epoch = 97
validation_samples_per_epoch = 22


model.compile(loss='mse', optimizer='adam')

model.fit_generator(training_generator, samples_per_epoch=samples_per_epoch,
                    nb_epoch=5, validation_data=validation_generator,
                    nb_val_samples=validation_samples_per_epoch)

print("Saving model weights and configuration file.")

model.save_weights('model.h5')
with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())




















