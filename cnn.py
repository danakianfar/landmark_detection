import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from keras.models import Model

landmark_dim = 2

# Load data
with open('all_data.pkl', 'rb') as f:
    images_train ,images_test, ldmks_2d_train , ldmks_2d_test , ldmks_3d_train , ldmks_3d_test , head_pose_train ,head_pose_test = pickle.load(f)

input_img = Input(shape=(3, 80, 120), name='input_img')
head_pose = Input(shape=(9,), name='head_pose')

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu', data_format="channels_first")(input_img)
#tower_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(tower_1)
#tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = Flatten()(tower_1)

x = keras.layers.concatenate([tower_1, head_pose], axis=1)
x = Dense(64, activation='relu')(x)
output = Dense(int(28 * landmark_dim), activation='relu', name='output')(x)


model = Model(inputs = [input_img, head_pose], outputs = [output])

model.compile(optimizer = 'adam', loss={'output': 'mean_squared_error'})

# And trained it via:
history = model.fit({'input_img': images_train, 'head_pose': head_pose_train},
          {'output': ldmks_2d_train},
          epochs=1, batch_size=32,
          validation_data=({'input_img': images_test, 'head_pose': head_pose_test}, {'output': ldmks_2d_test})
          )

plt.plot(history.history['loss'])

score = model.evaluate(({'input_img': images_test, 'head_pose': head_pose_test}, {'output': ldmks_2d_test}), verbose=0)
print('Test loss:', score[0])
