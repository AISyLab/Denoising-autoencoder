import os.path
import sys
import time
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
#from clr import OneCycleLR
import random
import matplotlib.pyplot as plot
import matplotlib.ticker as ticker
import os
import numpy as np
import warnings
from keras.models import load_model

from keras.callbacks import Callback
from keras import backend as K

# code based on https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/tree/master/ASCAD/N0%3D0
AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

def show(plot):
    plt.plot(plot)
    plt.show()

# Compute the position of the key hypothesis key amongst the hypotheses
def rk_key(rank_array,key):
    key_val = rank_array[key]
    return np.where(np.sort(rank_array)[::-1] == key_val)[0][0]


# Compute the evolution of rank
def rank_compute(prediction, att_plt, key, byte):
    """
    - prediction : predictions of the NN 
    - att_plt : plaintext of the attack traces 
    - key : Key used during encryption
    - byte : byte to attack    
    """
    
    (nb_trs, nb_hyp) = prediction.shape
    
    idx_min = nb_trs
    min_rk = 255
    
    key_log_prob = np.zeros(nb_hyp)
    rank_evol = np.full(nb_trs,255)
    prediction = np.log(prediction+1e-40)
                             
    for i in range(nb_trs):
        for k in range(nb_hyp):    
            key_log_prob[k] += prediction[i,AES_Sbox[k^att_plt[i,byte]]] #Computes the hypothesis values

        rank_evol[i] = rk_key(key_log_prob,key)

    return rank_evol


# Performs attack
def perform_attacks(nb_traces, predictions, nb_attacks, plt, key, byte=0, shuffle=True):
    """
    Performs a given number of attacks to be determined
    - nb_traces : number of traces used to perform the attack
    - predictions : array containing the values of the prediction
    - nb_attacks : number of attack to perform
    - plt : the plaintext used to obtain the consumption traces
    - key : the key used to obtain the consumption traces
    - byte : byte to attack
    - shuffle : (boolean, default = True)
    """

    (nb_total, nb_hyp) = predictions.shape

    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    for i in range(nb_attacks):
        if shuffle:
            l = list(zip(predictions,plt))
            random.shuffle(l)
            sp,splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt[:nb_traces]
        
        rank_evolution = rank_compute(att_pred,att_plt,key,byte=byte)
        all_rk_evol[i] = rank_evolution

    rk_avg = np.mean(all_rk_evol,axis=0)
    return rk_avg

def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

### CNN network
def cnn_architecture(input_size=700,learning_rate=0.00001,classes=256):
        
        # Designing input layer
        input_shape = (input_size,1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        
        x = Flatten(name='flatten')(x)

        # Classification layer
        x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
        x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
        
        # Logits layer              
        x = Dense(classes, activation='softmax', name='predictions')(x)

        # Create model
        inputs = img_input
        model = Model(inputs, x, name='ascad')
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        return model

def cnn_architecture_100(input_size=700,learning_rate=0.00001,classes=256):
        
        # Personal design
        input_shape = (input_size,1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        
        # 2nd convolutional block
        x = Conv1D(64, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(50, strides=50, name='block2_pool')(x)
        
        # 3rd convolutional block
        x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)      
        x = BatchNormalization()(x)
        x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
        
        x = Flatten(name='flatten')(x)

        # Classification part
        x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
        x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
        x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)

        # Logits layer
        x = Dense(classes, activation='softmax', name='predictions')(x)

        # Create model
        inputs = img_input
        model = Model(inputs, x, name='cnn_best')
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        return model

def cnn_best(length, lr=0.00001, classes=256):
    # From VGG16 design
    input_shape = (length, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    #optimizer = Adam(lr=learning_rate)
    optimizer = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def mlp_best(length, lr=0.00001, node=200, layer_nb=6):
    model = Sequential()
    model.add(Dense(node, input_dim=length, activation='relu'))
    for i in range(layer_nb - 2):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(256, activation='softmax'))
    optimizer = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

#### ASCAD helper to load profiling and attack data (traces and labels) (source : https://github.com/ANSSI-FR/ASCAD)
# Loads the profiling and attack datasets from the ASCAD database
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata']['plaintext'], in_file['Attack_traces/metadata']['plaintext'])

#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100, max_lr=1e-3):
    check_file_exists(os.path.dirname(save_file_name))
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    # reshape 2d to 3d if needed (in case attack with CNN)
    Reshaped_X_profiling, Reshaped_X_test = X_profiling, X_test
    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)
    callbacks=[save_model] 
           
    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), validation_data=(Reshaped_X_test, to_categorical(Y_test, num_classes=256)), batch_size=batch_size, verbose = 2, epochs=epochs)
    return history

if __name__ == "__main__":
    Denoised_data_dir = 'data.h5'
    model_name="data.h5"
    trained_model = "model.h5"

    # Choose the name of the model
    nb_epochs = 1000
    batch_size = 256
    input_size = 700
    learning_rate = 0.0001
    nb_traces_attacks = 10000
    nb_attacks = 100

    real_key = 224 # key for ascad fixed key
    real_key = 34 # key for ascad random key

    start = time.time()

    # Load the profiling traces
    (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(Denoised_data_dir, load_metadata=True)

    # Shuffle data
    #(X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

    # Choose your model
    model = cnn_best(len(X_profiling[0]))
    #model = mlp_best(len(X_profiling[0]))
    #model = load_model(trained_model)

    # Record the metrics
    history = train_model(X_profiling[10000:45000], Y_profiling[10000:45000], X_profiling[45000:], Y_profiling[45000:], model, ASCAD_trained_models_folder + model_name  + ".h5", epochs=nb_epochs, batch_size=batch_size, max_lr=learning_rate)
    end=time.time()
    print('Temps execution = %d'%(end-start))

    # Attack on the test traces
    predictions = model.predict(X_attack)
    avg_rank = np.array(perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, byte=2), int)
    
    print("\n t_GE = ")
    print(np.where(avg_rank<=0))
    np.save(Denoised_data_dir + '_GE_MLP', avg_rank)
    plt.plot(avg_rank)
    plt.savefig(Denoised_data_dir + '_GE_MLP.png')
    print(model_name)

