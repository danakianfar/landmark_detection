import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from keras.models import Model, load_model
from keras import backend as K


# Custom P-norm objective function
def p_norm_loss(y_true, y_pred):
    return K.mean(K.pow(y_pred - y_true, 4), axis=-1)


# Training parameters
batch_size = 32
epochs = 20
landmark_dim = 3

# Load data
with open('all_data.pkl', 'rb') as f:
    images_train, images_test, ldmks_2d_train, ldmks_2d_test, ldmks_3d_train, ldmks_3d_test, head_pose_train, head_pose_test, look_vec_train, look_vec_test = pickle.load(f)

# Define inputs for the network
input_img = Input(shape=(3, 80, 120), name='input_img')
head_pose = Input(shape=(9,), name='head_pose')

# Apply upper part of the network by doing convolutions
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(tower_1)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(tower_1)
tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = Flatten()(tower_1)
tower_1 = Dense(64, activation='relu')(tower_1)

# concatenate the output of the convolutions and the head_pose information
x = keras.layers.concatenate([tower_1, head_pose], axis=1)

# Pass the concatenated vector through some dense layers
x = Dense(64, activation='relu')(x)
output = Dense(int(28 * landmark_dim), activation='linear', name='output')(x)

# This model receives an input_img and a head_pose and returns output
# which is the landmarks of the dimension given by landmark_dim
model = Model(inputs = [input_img, head_pose], outputs = [output])

# Set optimizer and loss
model.compile(optimizer = 'adam', loss={'output': 'mean_squared_error'})

model.summary()

# Do the actual training
history = model.fit(
          {'input_img': images_train, 'head_pose': head_pose_train},
          {'output': ldmks_3d_train},
          epochs = epochs, batch_size= batch_size,
          validation_data = ({'input_img': images_test, 'head_pose': head_pose_test}, {'output': ldmks_3d_test})
          )

# Plot the evolution of the loss over time
plt.figure()
plt.plot(history.history['loss'])
plt.show()

# Save the learned model
model.save('landmark_cnn_3d.h5')

# Load the saved model
model = load_model('landmark_cnn_3d.h5', custom_objects={'p_norm_loss': p_norm_loss})

# Evaluate the model on the test data
score = model.evaluate(({'input_img': images_test, 'head_pose': head_pose_test}), {'output': ldmks_3d_test}, verbose=0)

# Print the loss on the test data
print('Test loss:', score)
