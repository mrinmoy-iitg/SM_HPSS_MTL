#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    [1] Doukhan, D., Lechapt, E., Evrard, M., & Carrive, J. (2018). Ina’s 
    mirex 2018 music and speech detection system. Music Information Retrieval 
    Evaluation eXchange (MIREX 2018).

    [2] Papakostas, M., & Giannakopoulos, T. (2018). Speech-music 
    discrimination using deep visual feature extractors. Expert Systems with 
    Applications, 114, 334-344.

    [3] Lemaire, Q., & Holzapfel, A. (2019). Temporal convolutional networks 
    for speech and music detection in radio broadcast. In 20th International 
    Society for Music Information Retrieval Conference, ISMIR 2019, 4-8 
    November 2019. International Society for Music Information Retrieval.

    [4] Jang, B. Y., Heo, W. H., Kim, J. H., & Kwon, O. W. (2019). Music 
    detection from broadcast contents using convolutional neural networks with 
    a Mel-scale kernel. EURASIP Journal on Audio, Speech, and Music Processing,
    2019(1), 1-12.

Created on Thu Jul 29 19:25:25 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

"""

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Input, Concatenate, MaxPooling2D, Activation, Dense, Flatten, Cropping2D, Lambda
from tensorflow.nn import local_response_normalization as LRN
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant, VarianceScaling, Zeros
from tensorflow.compat.v1.keras.initializers import RandomNormal
import numpy as np
import librosa




def get_Doukhan_model(PARAMS, n_classes=2):
    '''
    CNN architecture proposed by Doukhan et al. [1]

    Parameters
    ----------
    PARAMS : dict
        Contains default parameters.
    n_classes : int, optional
        Number of classes. Default is 2.

    Returns
    -------
    model : tensorflow.keras.models.Model
        CNN model.
    learning_rate : float
        Initial learning rate.

    '''
    k_init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
    b_init = Zeros()
    
    input_img = Input(PARAMS['input_shape'][PARAMS['Model']]) # (21 x 68 x 1)

    x = Conv2D(64, input_shape=PARAMS['input_shape'][PARAMS['Model']], kernel_size=(4, 5), strides=(1, 1), padding='valid', activation='linear', kernel_initializer=k_init, bias_initializer=b_init)(input_img) # matched with original
    x = BatchNormalization(axis=-1)(x) # matched with original
    x = Activation('relu')(x) # matched with original
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x) # matched with original
    
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear', kernel_initializer=k_init, bias_initializer=b_init)(x) # matched with original
    x = BatchNormalization(axis=-1)(x) # matched with original
    x = Activation('relu')(x) # matched with original

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear', kernel_initializer=k_init, bias_initializer=b_init)(x) # matched with original
    x = BatchNormalization(axis=-1)(x) # matched with original
    x = Activation('relu')(x) # matched with original
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x) # matched with original
    
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='linear', kernel_initializer=k_init, bias_initializer=b_init)(x) # matched with original
    x = BatchNormalization(axis=-1)(x) # matched with original
    x = Activation('relu')(x) # matched with original
    x = MaxPooling2D(pool_size=(1, 12), strides=(1, 12), padding='valid')(x) # matched with original
    
    x = Flatten()(x) # matched with original

    x = Dense(512, activation='linear', kernel_initializer=k_init, bias_initializer=b_init)(x) # matched with original
    x = BatchNormalization(axis=-1)(x) # matched with original
    x = Activation('relu')(x) # matched with original
    x = Dropout(0.2)(x) # matched with original

    x = Dense(512, activation='linear', kernel_initializer=k_init, bias_initializer=b_init)(x) # matched with original
    x = BatchNormalization(axis=-1)(x) # matched with original
    x = Activation('relu')(x) # matched with original
    x = Dropout(0.3)(x) # matched with original

    x = Dense(512, activation='linear', kernel_initializer=k_init, bias_initializer=b_init)(x) # matched with original
    x = BatchNormalization(axis=-1)(x) # matched with original
    x = Activation('relu')(x) # matched with original
    x = Dropout(0.4)(x) # matched with original
    
    x = Dense(512, activation='linear', kernel_initializer=k_init, bias_initializer=b_init)(x) # matched with original
    x = BatchNormalization(axis=-1)(x) # matched with original
    x = Activation('relu')(x) # matched with original
    x = Dropout(0.5)(x) # matched with original

    output = Dense(n_classes, activation='softmax', kernel_initializer=k_init, bias_initializer=b_init)(x)

    model = Model(input_img, output)
    learning_rate = 0.0001
    optimizer = optimizers.Adam(lr=learning_rate)
    
    if n_classes==2:
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])    
    elif n_classes==3:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])    
    
    print(model.summary())
    print('Architecture proposed by Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION\n')

    return model, learning_rate

        



def get_Papakostas_model(PARAMS, n_classes=2):    
    '''
    CNN architecture proposed by Papakostas et al. [2]    

    Parameters
    ----------
    PARAMS : dict
        Contains various parameters.
    n_classes : int, optional
        Number of classes. Default is 2.

    Returns
    -------
    model : tensorflow.keras.models.Model
        Cascaded MTL CNN model.
    learning_rate : float
        Initial learning rate.

    '''
    input_img = Input(PARAMS['input_shape'][PARAMS['Model']])

    x = Conv2D(96, input_shape=PARAMS['input_shape'][PARAMS['Model']], kernel_size=(5, 5), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(input_img)
    x = Lambda(lambda norm_lyr: LRN(norm_lyr, depth_radius=5, alpha=0.0001, beta=0.75))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
    x = Lambda(lambda norm_lyr: LRN(norm_lyr, depth_radius=5, alpha=0.0001, beta=0.75))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = Flatten()(x)
    
    x = Dense(4096, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(4096, kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(n_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)

    model = Model(input_img, output)

    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=700, decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule)
    
    if n_classes==2:
        model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])    
    elif n_classes==3:
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])    

    print(model.summary())
    print('Architecture proposed by Papakostas et al. Expert Systems with Applications 2018\n')

    return model, initial_learning_rate




def get_Lemaire_model(
        TR_STEPS, 
        # kernel_size=3, # Temporal Conv, MelSpec
        # Nd=8, # Temporal Conv, MelSpec
        # nb_stacks=3, # Temporal Conv, MelSpec
        # n_layers=1, # Temporal Conv, MelSpec
        # n_filters=32, # Temporal Conv, MelSpec
        # use_skip_connections=True, # Temporal Conv, MelSpec
        # ----
        kernel_size=3, # Temporal Conv, LogMelSpec
        Nd=8, # Temporal Conv, LogMelSpec
        nb_stacks=3, # Temporal Conv, LogMelSpec
        n_layers=1, # Temporal Conv, LogMelSpec
        n_filters=32, # Temporal Conv, LogMelSpec
        use_skip_connections=False, # Temporal Conv, LogMelSpec
        # ----
        activation='norm_relu', 
        bidirectional=True, 
        N_MELS=80, 
        n_classes=2, 
        patch_size=68,
        ):
    '''
    TCN based model architecture proposed by Lemaire et al. [3]
    Code source: https://github.com/qlemaire22/speech-music-detection    

    Parameters
    ----------
    TR_STEPS : int
        Number of training batches per epoch.
    n_filters : int, optional
        The default is 32.
    Nd : int, optional
        The default is 3.
    kernel_size : int, optional
        The default is 3.
    nb_stacks : int, optional
        The default is 10.
    activation : string, optional
        The default is 'norm_relu'.
    n_layers : int, optional
        The default is 3.
    use_skip_connections : boolean, optional
        The default is False.
    bidirectional : boolean, optional
        The default is True.
    N_MELS : int, optional
        The default is 80.
    n_classes : int, optional
        The default is 2.
    patch_size : int, optional
        The default is 68.

    Returns
    -------
    model : tensorflow.keras.models.Model
        CNN model.
    lr : float
        Learning rate.

    '''
    from tcn import TCN
    from tcn.tcn import process_dilations

    dilations = [2**nd for nd in range(Nd)]
    list_n_filters = [n_filters]*n_layers
    dropout_rate = np.random.uniform(0.05,0.5)
    bidirectional = True
    
    if bidirectional:
        padding = 'same'
    else:
        padding = 'causal'

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(patch_size, N_MELS))
        
    for i in range(n_layers):
        if i == 0:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, 'norm_relu', padding, use_skip_connections, dropout_rate, return_sequences=True)(input_layer)
        else:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, 'norm_relu', padding, use_skip_connections, dropout_rate, return_sequences=True, name="tcn" + str(i))(x)

    x = Flatten()(x)

    x = Dense(n_classes)(x)
    x = Activation('softmax')(x)
    output_layer = x
    
    model = Model(input_layer, output_layer)

    initial_learning_rate = 0.002
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=3*TR_STEPS, decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)

    if n_classes==2:
        model.compile(loss='binary_crossentropy', metrics='accuracy', optimizer=optimizer)
    elif n_classes==3:
        model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer=optimizer)

    print(model.summary())
    print('Architecture of Lemaire et. al. Proc. of the 20th ISMIR Conference, Delft, Netherlands, November 4-8, 2019\n')

    return model, initial_learning_rate




def get_kernel_initializer(w, m, t_dim=5):
    '''
    Generate the kernel initializer

    Parameters
    ----------
    w : list
        A specific Mel filter of size n_fft.
    m : list
        A list of size 2 that stores the indexes of the starting and ending 
        frequency bins.
    t_dim : int, optional
        Temporal dimension size of the Mel-scale kernel. The default is 5.

    Returns
    -------
    kernel_init : TYPE
        DESCRIPTION.

    '''
    kernel_init = w[m[0]:m[1]+1]
    kernel_init = np.repeat(np.array(kernel_init, ndmin=2).T, t_dim, axis=1)
    kernel_init = np.expand_dims(np.expand_dims(kernel_init, axis=2), axis=3)
    kernel_init = np.repeat(kernel_init, 3, axis=3)
    
    return kernel_init




def get_Jang_model(PARAMS, fs=16000, Tw=25, n_mels=64, t_dim=5, n_classes=2):
    '''
    Function to build the Mel-scale CNN architecture proposed by Jang et al. [4]
                
    Parameters
    ----------
    fs : int, optional
        Sampling rate. The default is 16000.
    Tw : int, optional
        Short-term frame size in miliseconds. The default is 25.
    n_mels : int, optional
        Number of mel-filters. The default is 64.
    t_dim : int, optional
        Time dimension of the mel-scale kernels. The default is 5.
    n_classes : int, optional
        Number of classes. The default is 2.

    Returns
    -------
    model : tensorflow.keras.models.Model
        Mel-scale CNN model.

    '''
    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    M = librosa.filters.mel(fs, n_fft=n_fft, n_mels=n_mels, norm='slaney')
    # print('M: ', np.shape(M), n_fft, n_mels)
    filter_bins = np.zeros((np.shape(M)[0],2))
    for i in range(np.shape(M)[0]):
        filter_bins[i,0] = np.squeeze(np.where(M[i,:]>0))[0]
        filter_bins[i,1] = np.squeeze(np.where(M[i,:]>0))[-1]
    filter_bins = filter_bins.astype(int)
    n_fft = np.shape(M)[1]
    # print('M: ', np.shape(M), np.shape(filter_bins), n_fft, n_mels)

    ''' melCL layer '''
    inp_img = Input(PARAMS['input_shape'][PARAMS['Model']]) # Input(shape=(n_fft, 101, 1))
    top = filter_bins[0,0]
    bottom = n_fft-filter_bins[0,1]-1
    inp_mel = Cropping2D(cropping=((top, bottom), (0, 0)))(inp_img)
    kernel_width = filter_bins[0,1]-filter_bins[0,0]+1
    kernel_init = get_kernel_initializer(M[0,:], filter_bins[0,:], t_dim)
    melCl = Conv2D(3, kernel_size=(kernel_width,t_dim), strides=(kernel_width,1), padding='same', name='melCl0', kernel_initializer=Constant(kernel_init), use_bias=False, kernel_regularizer=l1_l2())(inp_mel)
    # melCl = Activation('tanh')(melCl)
    # print('melCl: ', filter_bins[0,:], top, bottom, K.int_shape(inp_mel), K.int_shape(melCl))
    # print('melCl: ', filter_bins[0,:], kernel_width)

    for mel_i in range(1, n_mels):
        top = filter_bins[mel_i,0]
        bottom = n_fft-filter_bins[mel_i,1]-1
        inp_mel = Cropping2D(cropping=((top, bottom), (0, 0)))(inp_img)
        kernel_width = filter_bins[mel_i,1]-filter_bins[mel_i,0]+1
        kernel_init = get_kernel_initializer(M[mel_i,:], filter_bins[mel_i,:], t_dim)
        melCl_n = Conv2D(3, kernel_size=(kernel_width,t_dim), strides=(kernel_width,1), padding='same', name='melCl'+str(mel_i), kernel_initializer=Constant(kernel_init), use_bias=False, kernel_regularizer=l1_l2())(inp_mel)
        # melCl_n = Activation('tanh')(melCl_n)
        melCl = Concatenate(axis=1)([melCl, melCl_n])
        # print('melCl: ', filter_bins[mel_i,:], top, bottom, K.int_shape(inp_mel), K.int_shape(melCl))
        # print('melCl: ', filter_bins[mel_i,:], kernel_width)
    ''' ~~~~~~~~~~~~~~~~~~~~ '''
    # melCl = BatchNormalization(axis=-1)(melCl)
    melCl = Activation('tanh')(melCl)
    # melCl = Activation('sigmoid')(melCl)
    # melCl = Dropout(0.4)(melCl)
    # print('melCl: ', K.int_shape(melCl))
    
    x = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same')(melCl)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    # print('x1: ', K.int_shape(x))

    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    # print('x2: ', K.int_shape(x))

    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    # print('x3: ', K.int_shape(x))

    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    # print('x4: ', K.int_shape(x))

    x = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    # print('x5: ', K.int_shape(x))

    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    # print('x6: ', K.int_shape(x))
    
    x = Flatten()(x)
    # print('x7: ', K.int_shape(x))
    
    # Best results obtained without the FC layers
    # x = Dense(2048)(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.4)(x)
    # print('x8: ', K.int_shape(x))

    # x = Dense(1024)(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.4)(x)
    # print('x9: ', K.int_shape(x))
    
    output = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inp_img, output)
    
    learning_rate = 0.001
    
    if n_classes==2:
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    elif n_classes==3:
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    print(model.summary())
    print('Mel-scale CNN architecture proposed by Jang et al.')

    return model, learning_rate





# def get_Jang_model(PARAMS, fs=16000, Tw=25, n_mels=64, t_dim=5, n_classes=2):
#     '''
#     Function to build the Mel-scale CNN architecture proposed by Jang et al. [4]
                
#     Parameters
#     ----------
#     fs : int, optional
#         Sampling rate. The default is 16000.
#     Tw : int, optional
#         Short-term frame size in miliseconds. The default is 25.
#     n_mels : int, optional
#         Number of mel-filters. The default is 64.
#     t_dim : int, optional
#         Time dimension of the mel-scale kernels. The default is 5.
#     n_classes : int, optional
#         Number of classes. The default is 2.

#     Returns
#     -------
#     model : tensorflow.keras.models.Model
#         Mel-scale CNN model.

#     '''
#     n_fft = PARAMS['n_fft'][PARAMS['Model']]
#     M = librosa.filters.mel(fs, n_fft=n_fft, n_mels=n_mels, norm='slaney')
#     filter_bins = np.zeros((np.shape(M)[0],2))
#     for i in range(np.shape(M)[0]):
#         filter_bins[i,0] = np.squeeze(np.where(M[i,:]>0))[0]
#         filter_bins[i,1] = np.squeeze(np.where(M[i,:]>0))[-1]
#     filter_bins = filter_bins.astype(int)
#     n_fft = np.shape(M)[1]

#     ''' melCL layer '''
#     inp_img = Input(PARAMS['input_shape'][PARAMS['Model']]) # Input(shape=(n_fft, 101, 1))
#     top = filter_bins[0,0]
#     bottom = n_fft-filter_bins[0,1]
#     inp_mel = Cropping2D(cropping=((top, bottom), (0, 0)))(inp_img)
#     kernel_width = filter_bins[0,1]-filter_bins[0,0]+1
#     kernel_init = get_kernel_initializer(M[0,:], filter_bins[0,:], t_dim)
#     melCl = Conv2D(3, kernel_size=(kernel_width,t_dim), strides=(kernel_width,1), padding='same', name='melCl0', kernel_initializer=Constant(kernel_init), use_bias=True, kernel_regularizer=l2())(inp_mel)
#     # melCl = Activation('tanh')(melCl)

#     for mel_i in range(1, n_mels):
#         top = filter_bins[mel_i,0]
#         bottom = n_fft-filter_bins[mel_i,1]
#         inp_mel = Cropping2D(cropping=((top, bottom), (0, 0)))(inp_img)
#         kernel_width = filter_bins[mel_i,1]-filter_bins[mel_i,0]+1
#         kernel_init = get_kernel_initializer(M[mel_i,:], filter_bins[mel_i,:], t_dim)
#         melCl_n = Conv2D(3, kernel_size=(kernel_width,t_dim), strides=(kernel_width,1), padding='same', name='melCl'+str(mel_i), kernel_initializer=Constant(kernel_init), use_bias=True, kernel_regularizer=l2())(inp_mel)
#         # melCl_n = Activation('tanh')(melCl_n)
#         melCl = Concatenate(axis=1)([melCl, melCl_n])
#     ''' ~~~~~~~~~~~~~~~~~~~~ '''
#     melCl = BatchNormalization(axis=-1)(melCl)
#     melCl = Activation('tanh')(melCl)
#     melCl = Dropout(0.4)(melCl)
    
#     x = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2())(melCl)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)

#     x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)

#     x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2())(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)

#     x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)

#     x = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2())(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)

#     x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
    
#     x = Flatten()(x)
    
#     x = Dense(2048, kernel_regularizer=l2())(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)

#     x = Dense(1024, kernel_regularizer=l2())(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.4)(x)
    
#     output = Dense(n_classes, activation='softmax', kernel_regularizer=l2())(x)
    
#     model = Model(inp_img, output)
    
#     learning_rate = 0.001
    
#     if n_classes==2:
#         model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
#     elif n_classes==3:
#         model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

#     print(model.summary())
#     print('Mel-scale CNN architecture proposed by Jang et al.')

#     return model, learning_rate
