import os
import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.normalization import BatchNormalization
from keras.utils import multi_gpu_model, to_categorical
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K

Sbox = np.array([
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

####################### Utility functions #######################
def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def show(plot):
    plt.plot(plot)
    plt.show()

def load_ascad(ascad_database_file):
    check_file_exists(ascad_database_file)
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces  
    X_profiling = np.array(in_file['Profiling_traces/traces'], int)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    print('Profiling trace shape: ', np.shape(X_profiling))
    # Load attack traces  
    X_attack = np.array(in_file['Attack_traces/traces'], int)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    print('Attack trace shape: ', np.shape(X_attack))
    all_traces = np.concatenate((X_profiling, X_attack), axis = 0)
    return all_traces

def scale(v):
    return (v - v.min()) / (v.max() - v.min())

def unscale(o, v):
    return o * (v.max() - v.min()) + v.min()

# utility function for SNR calculation
def CalculateSNR(l, IntermediateData):
    trace_length = l.shape[1]
    mean = np.zeros([256, trace_length])
    var = np.zeros([256, trace_length])
    cpt = np.zeros(256)
    i = 0

    for trace in l:
        # classify the traces based on its SBox output
        # then add the classified traces together
        mean[IntermediateData[i]] += trace
        var[IntermediateData[i]] += np.square(trace)
        # count the trace number for each SBox output
        cpt[IntermediateData[i]] += 1
        i += 1

    for i in range(256):
        # average the traces for each SBox output
        mean[i] = mean[i] / cpt[i]
        # variance  = mean[x^2] - (mean[x])^2
        var[i] = var[i] / cpt[i] - np.square(mean[i])
    # Compute mean [var_cond] and var[mean_cond] for the conditional variances and means previously processed
    # calculate the trace variation in each points
    varMean = np.var(mean, 0)
    # evaluate if current point is stable for all S[p^k] cases
    MeanVar = np.mean(var, 0)
    return (varMean, MeanVar)

 # Calculate SNR
def SNR(trace_directory):
    # here, enter your own datapath
    data_dir = trace_dir = trace_directory
    n_samples = 10000
    targeted_sbox_index = 2
    targeted_keyExpansion_index = 42

    print("Read Traces, plaintexts, keys, masks values and initialize the labels used afterwards")
    f = h5py.File(trace_dir, "r")
    trace = h5py.File(trace_dir, "r")
    l = trace['Attack_traces/traces'][:n_samples, :]
    l = l.astype(float)

    data = np.array(f['Attack_traces/metadata'][:n_samples])
    k = data['key'][:, targeted_sbox_index]
    p = data['plaintext'][:, targeted_sbox_index]
    r = data['masks'][:, targeted_sbox_index-2]
    rout = data['masks'][:, 15]

    print("Calculate intermediate data")
    HW = np.array([bin(n).count("1") for n in range(0, 256)])
    SboxOut_withMaskRout = Sbox[k ^ p] ^ rout
    hw_SboxOut_withMaskRout = HW[SboxOut_withMaskRout]
    SboxOut_withoutMaskRout = SboxOut_withMaskRout ^ rout
    hw_SboxOut_withoutMaskRout = HW[SboxOut_withoutMaskRout]
    SboxOut_withMaskR = Sbox[k ^ p] ^ r

    plt.set_cmap('Blues')
    plt.tight_layout()

    print("Calculate SNR and plot the data")
    #IntermediateData = [SboxOut_withMaskRout, SboxOut_withoutMaskRout, SboxOut_withMaskR, rout, r]
    #FigureLable = ['SboxOut_withMaskRout', 'SboxOut_withoutMaskRout', 'SboxOut_withMaskR', 'rout', 'r']
    IntermediateData = [SboxOut_withMaskRout]
    FigureLable = ['SboxOut_withMaskRout']
    FigureArray = []
    for idx in range(len(IntermediateData)):
        varMean, MeanVar = CalculateSNR(l, IntermediateData[idx])
        snr, = plt.plot((varMean / MeanVar), label=FigureLable[idx])
        FigureArray.append(snr)
    plt.xlabel('Time Samples')
    plt.legend(handles=FigureArray, loc=2)
    plt.show()
    plt.savefig(trace_directory + '.png')

def padding(X, Y):
    X_trace_length = len(X[0])
    Y_trace_length = len(Y[0])
    diff = int(abs(X_trace_length - Y_trace_length) / 2)
    if diff == 0:
      print("No difference, returned!")
      return X
    output = np.zeros((len(Y), X_trace_length, 1))
    if diff != 0:
        for i in range(len(Y)):
            output[i][:, 0] = np.pad(Y[i][:, 0], (0, diff*2), 'constant')
    return output

def noiseFilter(input, order, model):
    filtered_imgs = model.predict(input)
    for i in range(order):
        filtered_imgs = model.predict(filtered_imgs)
    return np.array(filtered_imgs)

def generate_traces(model, labeled_traces_file, trace_data, trace_data_o):
    # Open the output labeled file for writing
    try:
        out_file = h5py.File(labeled_traces_file, "r+")
    except:
        print("Error: can't open HDF5 file '%s' for writing ..." % labeled_traces_file)
        sys.exit(-1)

    # denoise and unscale the traces
    all_traces = unscale(noiseFilter(trace_data, 0, model)[:,:,0], trace_data_o)

    print("Processing profiling traces...")
    profiling_traces = all_traces[:50000]
    print("Profiling: after")
    del out_file['Profiling_traces/traces']
    out_file.create_dataset('Profiling_traces/traces', data=np.array(profiling_traces, dtype=np.int8))

    print("Processing attack traces...")
    attack_traces = all_traces[50000:]
    print("Attack: after")
    del out_file['Attack_traces/traces']
    out_file.create_dataset('Attack_traces/traces',  data=np.array(attack_traces, dtype=np.int8))

    out_file.close()

    print("File integrity checking...")
    out_file = h5py.File(labeled_traces_file, 'r')
    profiling = out_file['Profiling_traces/traces']
    print(np.shape(profiling))
    print(np.allclose(profiling[()], profiling_traces))
    attack = out_file['Attack_traces/traces']
    print(np.allclose(attack[()], attack_traces))
    out_file.close()
    print("Done!") 

####################### Denoiser building blocks #######################
def conv(x, filter_num, window_size, act, max_pool, dp_rate = 0):
  y = Conv1D(filter_num, window_size, padding='same')(x)
  y = BatchNormalization()(y)
  y = Activation(act)(y)
  if max_pool > 0:
    y = MaxPooling1D(max_pool)(y)
  if dp_rate > 0:
    y = Dropout(dp_rate)(y)
  return y

def Conv1DTranspose(input_tensor, filters, kernel_size, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def deconv(x, filter_num, window_size, act, max_pool, dp_rate = 0):
  if max_pool > 0:
    y = UpSampling1D(max_pool)(x)
  else:
    y = x
  y = Conv1DTranspose(y, filter_num, window_size)
  y = BatchNormalization()(y)
  y = Activation(act)(y)

  if dp_rate > 0:
    y = Dropout(dp_rate)(y)
  return y

def DAE(lr, input_length):
    img_input = Input(shape=(input_length, 1))
    #encoder
    x = conv(img_input, 256, 2, 'selu', 0)
    x = conv(x, 256, 2, 'selu', 0)
    x = conv(x, 256, 2, 'selu', 0)
    x = conv(x, 256, 2, 'selu', 0)
    x = conv(x, 256, 2, 'selu', 0)
    x = conv(x, 256, 2, 'selu', 5)
    x = conv(x, 128, 2, 'selu', 0)
    x = conv(x, 128, 2, 'selu', 0)
    x = conv(x, 128, 2, 'selu', 0)
    x = conv(x, 128, 2, 'selu', 2)
    x = conv(x, 64, 2, 'selu', 0)
    x = conv(x, 64, 2, 'selu', 0)
    x = conv(x, 64, 2, 'selu', 0)
    x = conv(x, 64, 2, 'selu', 2)
    x = Flatten(name='flatten')(x)

    x = Dense(512, activation='selu')(x)

    x = Dense(2240, activation='selu')(x)
    x = Reshape((35, 64))(x)
    x = deconv(x, 64, 2, 'selu', 2)
    x = deconv(x, 64, 2, 'selu', 0)
    x = deconv(x, 64, 2, 'selu', 0)
    x = deconv(x, 64, 2, 'selu', 0)
    x = deconv(x, 128, 2, 'selu', 2)
    x = deconv(x, 128, 2, 'selu', 0)
    x = deconv(x, 128, 2, 'selu', 0)
    x = deconv(x, 128, 2, 'selu', 0)
    x = deconv(x, 256, 2, 'selu', 5)
    x = deconv(x, 256, 2, 'selu', 0)
    x = deconv(x, 256, 2, 'selu', 0)
    x = deconv(x, 256, 2, 'selu', 0)
    x = deconv(x, 256, 2, 'selu', 0)
    x = deconv(x, 256, 2, 'selu', 0)
    
    x = deconv(x, 1, 2, 'sigmoid', 0)

    model = Model(img_input, x)
    #model = multi_gpu_model(model)
    opt = RMSprop(lr=lr)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
  


if __name__ == "__main__":
    noisy_traces_dir = "Noisy.h5"
    clean_trace_dir = "Clean.h5"
    denoised_traces_dir = "Denoised.h5"
    model_dir = "model.h5"

    # load traces
    print('load traces...')
    X_noisy_o = load_ascad(noisy_traces_dir)
    Y_clean_o = load_ascad(clean_trace_dir)
    X_noisy = scale(X_noisy_o)
    Y_clean = scale(Y_clean_o) 
    
    print('train model...')
    autoencoder = DAE(0.0001, len(X_noisy[0]))
    #autoencoder = load_model(model_dir)

    # save_model = ModelCheckpoint(model_dir)
    # callbacks = [save_model]

    autoencoder.fit(X_noisy[:10000], Y_clean[:10000], 
                    validation_data=(X_noisy[45000:], Y_clean[45000:]),
                    epochs=100,
                    batch_size=128,
                    verbose=2)

    print('generate traces...')
    generate_traces(autoencoder, denoised_traces_dir, X_noisy, Y_clean_o)
    SNR(denoised_traces_dir)

