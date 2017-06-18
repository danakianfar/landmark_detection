import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from keras.models import Model, load_model
from keras import backend as K
import os

# Supress redundant tf logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Custom P-norm objective function
def p_norm_loss(y_true, y_pred):
    return K.mean(K.pow(y_pred - y_true, 4), axis=-1)

def landmark_accuracy(y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred) < 3.)

# Compile model
def define_network_architecture(landmark_dim = 2, use_headpose = True, conv_type="non_spatial", double_tower=False, loss_function = 'mean_squared_error'):

    print('Defining network architecture...')

    # Define inputs for the network
    input_img = Input(shape=(3, 80, 120), name='input_img')
    head_pose = Input(shape=(9,), name='head_pose')

    if conv_type == 'non_spatial':
        data_format = 'channels_last'
    else:
        data_format = 'channels_first'

    # Apply upper part of the network by doing convolutions
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format=data_format)(input_img)
    tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(tower_1)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format=data_format)(tower_1)
    tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(tower_1)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu', data_format=data_format)(tower_1)
    x = Flatten()(tower_1)
    x = Dense(64, activation='relu')(x)

    if double_tower:
        # Apply upper part of the network by doing convolutions
        tower_2 = Conv2D(16, (3, 3), padding='valid', activation='relu', data_format = 'channels_first')(input_img)
        tower_2 = MaxPooling2D((2, 2), strides=(2, 2))(tower_2)
        tower_2 = Conv2D(16, (3, 3), padding='valid', activation='relu', data_format = 'channels_first')(tower_2)
        tower_2 = MaxPooling2D((3, 3), strides=(3, 3))(tower_2)
        tower_2 = Flatten()(tower_2)

        # concatenate the output of the convolutions and the head_pose information
        if use_headpose:
            x = keras.layers.concatenate([x, tower_2, head_pose], axis=1)
        else:
            x = keras.layers.concatenate([x, tower_2], axis=1)

    elif use_headpose:
        # concatenate the output of the convolutions and the head_pose information
        x = keras.layers.concatenate([x, head_pose], axis=1)
        
    # Pass the concatenated vector through a dense layer
    x = Dense(64, activation='relu')(x)
    
    # Output
    output = Dense(int(28 * landmark_dim), activation='linear', name='output')(x)

    # This model receives an input_img and a head_pose and returns output
    # which is the landmarks of the dimension given by landmark_dim
    model = Model(inputs = [input_img, head_pose], outputs = [output])

    # Set optimizer and loss
    if loss_function == 'p_norm_loss':
        loss_function = p_norm_loss

    metrics = []
    if landmark_dim == 2: 
        metrics.append(landmark_accuracy)

    model.compile(optimizer = 'adam', loss={'output': loss_function}, metrics = metrics)

    return model

def train_model(model, images_train, head_pose_train, landmarks_train, 
    images_test, head_pose_test, landmarks_test, batch_size = 32, epochs = 30, save_name='landmark_unnamed', use_early_stopping=True):

    print('Training model %s' % save_name)

    delta = 1e-6
    if use_early_stopping:
        delta = 0

    # Early stopping delta < 1e-5
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=delta, verbose=1, mode='auto')]

    # Do the actual training
    history = model.fit(
              {'input_img': images_train, 'head_pose': head_pose_train},
              {'output': landmarks_train},
              epochs = epochs, batch_size= batch_size,
              validation_data = ({'input_img': images_test, 'head_pose': head_pose_test}, {'output': landmarks_test}),
              shuffle=True, 
              verbose=1, 
              callbacks=callbacks
              ).history

    # Evaluate the model on the test data
    score = model.evaluate(({'input_img': images_test, 'head_pose': head_pose_test}), {'output': landmarks_test}, verbose=0)

    # Print the loss on the test data
    print('Model %s \nTest loss: %s \n' % (save_name, score))

    # Save the learned model
    model.save('models/%s-%s.h5' % (save_name, score))

    with open('models/%s-%s.history' % (save_name, score), 'wb') as f:
        pickle.dump(history, f)

    return history, model




# Intiailize training parameters
batch_size = 64
epochs = 50

# # Load data
with open('all_data.pkl', 'rb') as f:
    images_train, images_test, ldmks_2d_train, ldmks_2d_test, ldmks_3d_train, ldmks_3d_test, head_pose_train, head_pose_test, look_vec_train, look_vec_test = pickle.load(f)

# Run a grid of experiments
for data_format in ['non_spatial']:
    for double_tower in [False, True]:
        for use_headpose in [True, False]: # whether to use headpose
            for landmark_dim in [2,3]: # 2D or 3D prediction
                for loss_function in ['mean_squared_error', 'mean_absolute_error', 'p_norm_loss']: # objective functions
                    # File name for saving
                    save_name = 'Head%s-Tower%s-%sD-%s-%s' % (str(use_headpose), str(double_tower), str(landmark_dim), loss_function, data_format)

                    # Get model
                    model = define_network_architecture(landmark_dim, use_headpose, data_format, double_tower, loss_function)

                    model.summary()

                    # Choose appropriate targets
                    if landmark_dim == 2:
                        landmarks_train = ldmks_2d_train
                        landmarks_test = ldmks_2d_test
                    else:
                        landmarks_train = ldmks_3d_train
                        landmarks_test = ldmks_3d_test

                    # Use early stopping when using double tower architecture (spatial + non-spatial)
                    use_early_stopping = double_tower

                    # Train model
                    history, model = train_model(model, images_train, head_pose_train, landmarks_train, images_test, 
                        head_pose_test, landmarks_test, batch_size = batch_size, epochs = epochs, save_name=save_name, use_early_stopping=use_early_stopping)


# # Training parameters
# batch_size = 32
# epochs = 20
# landmark_dim = 3

# # Load data
# with open('all_data.pkl', 'rb') as f:
#     images_train, images_test, ldmks_2d_train, ldmks_2d_test, ldmks_3d_train, ldmks_3d_test, head_pose_train, head_pose_test, look_vec_train, look_vec_test = pickle.load(f)

# # Define inputs for the network
# input_img = Input(shape=(3, 80, 120), name='input_img')
# head_pose = Input(shape=(9,), name='head_pose')

# # Apply upper part of the network by doing convolutions
# tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
# tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(tower_1)
# tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
# tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(tower_1)
# tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)
# tower_1 = Flatten()(tower_1)
# tower_1 = Dense(64, activation='relu')(tower_1)

# # concatenate the output of the convolutions and the head_pose information
# x = keras.layers.concatenate([tower_1, head_pose], axis=1)

# # Pass the concatenated vector through some dense layers
# x = Dense(64, activation='relu')(x)
# output = Dense(int(28 * landmark_dim), activation='linear', name='output')(x)

# # This model receives an input_img and a head_pose and returns output
# # which is the landmarks of the dimension given by landmark_dim
# model = Model(inputs = [input_img, head_pose], outputs = [output])

# # Set optimizer and loss
# model.compile(optimizer = 'adam', loss={'output': 'mean_squared_error'})

# model.summary()

# # Do the actual training
# history = model.fit(
#           {'input_img': images_train, 'head_pose': head_pose_train},
#           {'output': ldmks_3d_train},
#           epochs = epochs, batch_size= batch_size,
#           validation_data = ({'input_img': images_test, 'head_pose': head_pose_test}, {'output': ldmks_3d_test})
#           )

# # Plot the evolution of the loss over time
# plt.figure()
# plt.plot(history.history['loss'])
# plt.show()

# # Save the learned model
# model.save('landmark_cnn_3d.h5')

# # Load the saved model
# model = load_model('landmark_cnn_3d.h5', custom_objects={'p_norm_loss': p_norm_loss})

# # Evaluate the model on the test data
# score = model.evaluate(({'input_img': images_test, 'head_pose': head_pose_test}), {'output': ldmks_3d_test}, verbose=0)

# # Print the loss on the test data
# print('Test loss:', score)
