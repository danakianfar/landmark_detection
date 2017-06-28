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
    y_true = K.reshape(y_true, (-1, 28, 2))
    y_pred = K.reshape(y_pred, (-1, 28, 2))

    return K.mean( K.sum (K.abs(y_true - y_pred), axis=2) < 5.)

def landmark_accuracy_5(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred) < 5.)

def landmark_loss(y_true, y_pred):
    return K.mean( K.square(y_true - y_pred) * K.sigmoid( K.abs(y_true - y_pred) - 1 ), axis=-1)

def get_non_spatial_tower(input_img):

    data_format = 'channels_last' # retarded convolutions

    # Apply upper part of the network by doing convolutions
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format=data_format)(input_img)
    tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(tower_1)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu', data_format=data_format)(tower_1)
    tower_1 = MaxPooling2D((2, 2), strides=(1, 1))(tower_1)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu', data_format=data_format)(tower_1)
    tower_1 = Flatten()(tower_1)
    tower_1 = Dense(64, activation='relu')(tower_1)

    return tower_1

def get_spatial_tower(input_img):

    data_format = 'channels_first' #

    tower_2 = MaxPooling2D((2, 2), strides=(2, 2), name='spatial_pool0', data_format=data_format)(input_img)
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu', name='spatial_conv1', data_format = data_format)(tower_2)
    tower_2 = MaxPooling2D((2, 2), strides=(2, 2), name='spatial_pool1', data_format = data_format)(tower_2)
    tower_2 = Conv2D(48, (3, 3), padding='same', activation='relu', name='spatial_conv2', data_format = data_format)(tower_2)
    tower_2 = MaxPooling2D((2, 2), strides=(2, 2), name='spatial_pool2', data_format = data_format)(tower_2)
    tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='spatial_conv3', data_format = data_format)(tower_2)
    tower_2 = MaxPooling2D((2, 2), strides=(2, 2), name='spatial_pool3', data_format = data_format)(tower_2)
    tower_2 = Conv2D(64, (3, 3), padding='valid', activation='relu', name='spatial_conv4', data_format = data_format)(tower_2)
    # tower_2 = MaxPooling2D((2, 2), strides=(2, 2), name='spatial_pool4', data_format=data_format)(tower_2)
    tower_2 = Flatten()(tower_2)
    tower_2 = Dense(64, activation='relu')(tower_2)

    return tower_2

# Compile model
def define_network_architecture(landmark_dim = 2, use_headpose = True, topology= 'non_spatial', loss_function = 'mean_squared_error'):

    print('Defining network architecture...')

    # Define inputs for the network
    input_img = Input(shape=(1, 80, 120), name='input_img')
    inputs = [input_img]

    # If appending headpose information
    if use_headpose:
        head_pose = Input(shape=(9,), name='head_pose')
        inputs.append(head_pose)


    # If using both spatial and non-spatial convs
    if topology == 'double_tower':
        non_spatial = get_non_spatial_tower(input_img) # left tower
        spatial = get_spatial_tower(input_img) # right tower
        
        # concatenate the output of the convolutions, and the head_pose information (if necessary)
        if use_headpose:
            x = keras.layers.concatenate([non_spatial, spatial, head_pose], axis=1)
        else:
            x = keras.layers.concatenate([non_spatial, spatial], axis=1)

    else: # Single tower mode
        if topology == 'non_spatial':
            x = get_non_spatial_tower(input_img)

        elif topology == 'spatial': 
            x = get_spatial_tower(input_img)

        if use_headpose:
            # concatenate the output of the convolutions and the head_pose information
            x = keras.layers.concatenate([x, head_pose], axis=1)

        
    # Pass the concatenated vector through a dense layer
    # x = Dense(32, activation='relu')(x)
    
    # Output
    output = Dense(int(28 * landmark_dim), activation='linear', name='output')(x)

    # This model receives an input_img and a head_pose and returns output
    # which is the landmarks of the dimension given by landmark_dim
    model = Model(inputs = inputs, outputs = [output])

    # Parse loss function parameter
    if loss_function == 'p_norm_loss':
        loss_function = p_norm_loss
    elif loss_function == 'landmark_loss':
        loss_function = landmark_loss 

    # Metric for evaluation
    metrics = []
    if landmark_dim == 2: # Only 2D
        metrics.extend([landmark_accuracy])

    # Compile model
    adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
    model.compile(optimizer = adam, loss={'output': loss_function}, metrics = metrics)

    return model

def train_model(model, images_train, head_pose_train, landmarks_train, 
    images_test, head_pose_test, landmarks_test, batch_size = 32, epochs = 30, save_name='landmark_unnamed', use_early_stopping=True):

    print('Training model %s' % save_name)

    delta = 1e-6
    if not use_early_stopping:
        delta = 0

    # Early stopping delta < 1e-5
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=delta, verbose=1, mode='auto')]

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
    model.save('models/bw/%s-%s.h5' % (save_name, score))

    with open('models/bw/%s-%s.history' % (save_name, score), 'wb') as f:
        pickle.dump(history, f)

    return history, model




# Intiailize training parameters
batch_size = 32
epochs = 100

# # Load data
with open('all_data_augmented_bw.pkl', 'rb') as f:
    images_train, images_test, ldmks_2d_train, ldmks_2d_test, augmented_images_train, augmented_images_test, augmented_ldmks_2d_train, augmented_ldmks_2d_test, ldmks_3d_train, ldmks_3d_test, head_pose_train, head_pose_test, look_vec_train, look_vec_test = pickle.load(f)

# num_original_samples = len(images_train)
# rand_idx = np.random.choice(len(images_train), num_original_samples)
# X_train = np.vstack((images_train, augmented_images_train))
# X_test = np.vstack((images_test, augmented_images_test))

# Y_train = np.vstack((ldmks_2d_train, augmented_ldmks_2d_train))
# Y_test = np.vstack((ldmks_2d_test, augmented_ldmks_2d_test))

X_train = images_train
X_test = images_test

Y_train = ldmks_3d_train
Y_test = ldmks_3d_test


# Run a grid of experiments
for topology in ['spatial' ]:
    for use_headpose in [False]: # whether to use headpose
        for landmark_dim in [3]: # 2D or 3D prediction
            for loss_function in ['mean_squared_error']: # objective functions
                # File name for saving
                save_name = 'Head%s-%s-%sD-%s' % (str(use_headpose), topology, str(landmark_dim), loss_function)

                # Get model
                model = define_network_architecture(landmark_dim, use_headpose, topology, loss_function)

                model.summary()

                # Use early stopping when using double tower architecture (spatial + non-spatial)
                use_early_stopping = True

                # Train model
                history, model = train_model(model, X_train, head_pose_train, Y_train, X_test, 
                    head_pose_test, Y_test, batch_size = batch_size, epochs = epochs, save_name=save_name, use_early_stopping=use_early_stopping)


# batch_size = 32
# epochs = 1

# custom_objects={'p_norm_loss': p_norm_loss, 'landmark_accuracy' : landmark_accuracy, 'landmark_loss': landmark_loss}


# # Train pre-loaded model
# path = 'models/bw/HeadFalse-spatial-2D-mean_squared_error-[3.1770277053680553, 0.90926347746888847].h5'
# model = keras.models.load_model(path, custom_objects)
# save_name = 'HeadFalse-spatial-2D-mean_squared_error'

# # Reset weights
# # weights = model.get_weights()
# # weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
# # # Faster, but less random: only permutes along the first dimension
# # # weights = [np.random.permutation(w) for w in weights]
# # model.set_weights(weights)

# use_early_stopping = True

# # Train model
# history, model = train_model(model, X_train, head_pose_train, Y_train, X_test, 
#                     head_pose_test, Y_test, batch_size = batch_size, epochs = epochs, save_name=save_name, use_early_stopping=use_early_stopping)

