#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 13:10:13 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

"""

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Input, Concatenate, MaxPooling2D, Activation, Dense, Flatten, Cropping2D, Lambda
# import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.compat.v1.keras.initializers import RandomNormal
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from tensorflow.nn import local_response_normalization as LRN
from tensorflow.keras.initializers import VarianceScaling, Zeros
import librosa



def MTL_modifications(x):
    '''
    MTL modifications to be applied to the baseline models. The modified
    architecture is tuned on the MUSAN dataset.
    
    Parameters
    ----------
    x : Keras tensor
        Prefinal layer of an architecture to which MTL is to be applied.

    Returns
    -------
    sp_output : Keras tensor
        Output layer of the speech/non-speech task.
    mu_output : Keras tensor
        Output layer of the music/non-music task.
    smr_output : Keras tensor
        Output layer of the SMR regression task.

    '''
    #''' Speech/Non-Speech output '''
    x_sp = Dense(16, kernel_regularizer=l2())(x)
    x_sp = BatchNormalization(axis=-1)(x_sp)
    x_sp = Activation('relu')(x_sp)
    x_sp = Dropout(0.4)(x_sp)

    sp_output = Dense(1, activation='sigmoid', name='S')(x_sp)


    #''' Music/Non-Music output '''
    x_mu = Dense(16, kernel_regularizer=l2())(x)
    x_mu = BatchNormalization(axis=-1)(x_mu)
    x_mu = Activation('relu')(x_mu)
    x_mu = Dropout(0.4)(x_mu)

    x_mu = Dense(16, kernel_regularizer=l2())(x)
    x_mu = BatchNormalization(axis=-1)(x_mu)
    x_mu = Activation('relu')(x_mu)
    x_mu = Dropout(0.4)(x_mu)
    
    mu_output = Dense(1, activation='sigmoid', name='M')(x_mu)
    
    #''' Speech-to-Music Ratio (SMR) output '''
    x_smr = Dense(16, kernel_regularizer=l2())(x)
    x_smr = BatchNormalization(axis=-1)(x_smr)
    x_smr = Activation('relu')(x_smr)
    x_smr = Dropout(0.4)(x_smr)

    x_smr = Dense(16, kernel_regularizer=l2())(x)
    x_smr = BatchNormalization(axis=-1)(x_smr)
    x_smr = Activation('relu')(x_smr)
    x_smr = Dropout(0.4)(x_smr)
    
    smr_output = Dense(2, activation='linear', name='R')(x_smr)
    
    return sp_output, x_sp, mu_output, x_mu, smr_output, x_smr




def get_Lemaire_MTL_model(
        TR_STEPS,
        N_MELS=120,
        n_classes=3,
        patch_size=68,
        loss_weights=None, #{'S': 1.0, 'M': 1.0, 'R': 1.0, '3C': 1.0},
        ):
    '''
    MTL modification of the TCN based model architecture proposed by 
    Lemaire et al. [3]
    Code source: https://github.com/qlemaire22/speech-music-detection
    The model parameters are tuned on the MUSAN dataset.

        [3] Lemaire, Q., & Holzapfel, A. (2019). Temporal convolutional networks 
    for speech and music detection in radio broadcast. In 20th International 
    Society for Music Information Retrieval Conference, ISMIR 2019, 4-8 
    November 2019. International Society for Music Information Retrieval.

    Parameters
    ----------
    TR_STEPS : int
        Number of training batches per epoch.
    N_MELS : int, optional
        The default is 120.
    n_classes : int, optional
        The default is 3.
    patch_size : int, optional
        The default is 68.
    loss_weights : dict, optional
        The default is {'S': 1.0, 'M': 1.0, 'R': 1.0, '3C': 1.0}.

    Returns
    -------
    model : tensorflow.keras.models.Model
        CNN model.
    lr : float
        Learning rate.

    '''
    from tcn import TCN
    from tcn.tcn import process_dilations

    kernel_size = 3 # Temporal Conv, LogMelSpec
    Nd = 8 # Temporal Conv, LogMelSpec
    nb_stacks = 3 # Temporal Conv, LogMelSpec
    n_layers = 1 # Temporal Conv, LogMelSpec
    n_filters = 32 # Temporal Conv, LogMelSpec
    use_skip_connections = False # Temporal Conv, LogMelSpec
    activation = 'norm_relu' 
    dilations = [2**nd for nd in range(Nd)]
    list_n_filters = [n_filters]*n_layers
    dropout_rate = np.random.uniform(0.05,0.5)
    padding = 'same'
    dilations = process_dilations(dilations)
    
    input_layer = Input(shape=(patch_size,N_MELS))
        
    for i in range(n_layers):
        if i == 0:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True)(input_layer)
        else:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True, name="tcn" + str(i))(x)

    x = Flatten()(x)

    classification_output = Dense(n_classes, activation='softmax', name='3C')(x)

    sp_output, x_sp, mu_output, x_mu, smr_output, x_smr = MTL_modifications(x)
    
    model = Model(input_layer, [sp_output, mu_output, smr_output, classification_output])

    initial_learning_rate = 0.002
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=3*TR_STEPS, decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)

    model.compile(
        loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'},
        loss_weights=loss_weights,
        optimizer=optimizer, 
        metrics={'3C':'accuracy'}
        )

    print(model.summary())
    print('MTL modification of the architecture of Lemaire et. al. Proc. of the 20th ISMIR Conference, Delft, Netherlands, November 4-8, 2019\n')

    return model, initial_learning_rate




def cascade_MTL_modifications(x):
    '''
    Cascaded MTL modifications to be applied to the baseline models. The modified
    architecture is tuned on the MUSAN dataset.
    
    Parameters
    ----------
    x : Keras tensor
        Prefinal layer of an architecture to which MTL is to be applied.

    Returns
    -------
    sp_output : Keras tensor
        Output layer of the speech/non-speech task.
    mu_output : Keras tensor
        Output layer of the music/non-music task.
    smr_output : Keras tensor
        Output layer of the SMR regression task.

    '''
    #''' Speech-to-Music Ratio (SMR) output '''
    # x_smr = Dense(16, kernel_regularizer=l2())(x)
    # x_smr = BatchNormalization(axis=-1)(x_smr)
    # x_smr = Activation('relu')(x_smr)
    # x_smr = Dropout(0.4)(x_smr)

    x_smr = Dense(16, kernel_regularizer=l2())(x)
    x_smr = BatchNormalization(axis=-1)(x_smr)
    x_smr = Activation('relu')(x_smr)
    x_smr = Dropout(0.4)(x_smr)

    smr_output = Dense(2, activation='linear', name='R')(x_smr)

    #''' Speech/Non-Speech output '''
    x_sp = Dense(16, kernel_regularizer=l2())(x)
    x_sp = BatchNormalization(axis=-1)(x_sp)
    x_sp = Activation('relu')(x_sp)
    x_sp = Dropout(0.4)(x_sp)

    x_sp = Concatenate(axis=-1)([x_sp, smr_output])
    x_sp = BatchNormalization(axis=-1)(x_sp)

    sp_output = Dense(1, activation='sigmoid', name='S')(x_sp)


    #''' Music/Non-Music output '''
    # x_mu = Dense(16, kernel_regularizer=l2())(x)
    # x_mu = BatchNormalization(axis=-1)(x_mu)
    # x_mu = Activation('relu')(x_mu)
    # x_mu = Dropout(0.4)(x_mu)

    x_mu = Dense(16, kernel_regularizer=l2())(x)
    x_mu = BatchNormalization(axis=-1)(x_mu)
    x_mu = Activation('relu')(x_mu)
    x_mu = Dropout(0.4)(x_mu)

    x_mu = Concatenate(axis=-1)([x_mu, smr_output])
    x_mu = BatchNormalization(axis=-1)(x_mu)
    
    mu_output = Dense(1, activation='sigmoid', name='M')(x_mu)
            
    return sp_output, x_sp, mu_output, x_mu, smr_output, x_smr





def get_Lemaire_Cascaded_MTL_model(
        TR_STEPS,
        N_MELS=120,
        n_classes=3,
        patch_size=68,
        ):
    '''
    Cascaded MTL modification of the TCN based model architecture proposed by 
    Lemaire et al. [3]
    Code source: https://github.com/qlemaire22/speech-music-detection
    The model parameters are tuned on the MUSAN dataset.

        [3] Lemaire, Q., & Holzapfel, A. (2019). Temporal convolutional networks 
    for speech and music detection in radio broadcast. In 20th International 
    Society for Music Information Retrieval Conference, ISMIR 2019, 4-8 
    November 2019. International Society for Music Information Retrieval.

    Parameters
    ----------
    TR_STEPS : int
        Number of training batches per epoch.
    N_MELS : int, optional
        The default is 100.
    n_classes : int, optional
        The default is 3.
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

    kernel_size = 3 # Temporal Conv, LogMelSpec
    Nd = 8 # Temporal Conv, LogMelSpec
    nb_stacks = 3 # Temporal Conv, LogMelSpec
    n_layers = 1 # Temporal Conv, LogMelSpec
    n_filters = 32 # Temporal Conv, LogMelSpec
    use_skip_connections = False # Temporal Conv, LogMelSpec
    activation = 'norm_relu' 
    dilations = [2**nd for nd in range(Nd)]
    list_n_filters = [n_filters]*n_layers
    dropout_rate = np.random.uniform(0.05,0.5)
    padding = 'same'
    dilations = process_dilations(dilations)
    
    input_layer = Input(shape=(patch_size,N_MELS))
        
    for i in range(n_layers):
        if i == 0:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True)(input_layer)
        else:
            x = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True, name="tcn" + str(i))(x)

    x = Flatten()(x)

    sp_output, x_sp, mu_output, x_mu, smr_output, x_smr = cascade_MTL_modifications(x)
    
    classification_output = Dense(n_classes, activation='softmax', name='3C')(x)
    
    model = Model(input_layer, [sp_output, mu_output, smr_output, classification_output])

    initial_learning_rate = 0.002
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=3*TR_STEPS, decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)

    model.compile(
        loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'}, 
        optimizer=optimizer, 
        metrics={'3C':'accuracy'}
        )

    print(model.summary())
    print('Cascaded-MTL modification of the architecture of Lemaire et. al. Proc. of the 20th ISMIR Conference, Delft, Netherlands, November 4-8, 2019\n')

    return model, initial_learning_rate



def get_Lemaire_MTL_intermediate_fusion_model(
        TR_STEPS,
        N_MELS=120,
        n_classes=3,
        patch_size=68,
        ):
    '''
    TCN based model architecture proposed by Lemaire et al. [3]
    Code source: https://github.com/qlemaire22/speech-music-detection
    The model parameters are tuned on the MUSAN dataset.

        [3] Lemaire, Q., & Holzapfel, A. (2019). Temporal convolutional networks 
    for speech and music detection in radio broadcast. In 20th International 
    Society for Music Information Retrieval Conference, ISMIR 2019, 4-8 
    November 2019. International Society for Music Information Retrieval.

    Parameters
    ----------
    TR_STEPS : int
        Number of training batches per epoch.
    N_MELS : int, optional
        The default is 100.
    n_classes : int, optional
        The default is 3.
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

    kernel_size = 3 # Temporal Conv, LogMelSpec
    Nd = 8 # Temporal Conv, LogMelSpec
    nb_stacks = 3 # Temporal Conv, LogMelSpec
    n_layers = 1 # Temporal Conv, LogMelSpec
    n_filters = 32 # Temporal Conv, LogMelSpec
    use_skip_connections = False # Temporal Conv, LogMelSpec
    activation = 'norm_relu' 
    dilations = [2**nd for nd in range(Nd)]
    list_n_filters = [n_filters]*n_layers
    dropout_rate = np.random.uniform(0.05,0.5)
    padding = 'same'
    dilations = process_dilations(dilations)
    
    input_layer_H = Input(shape=(patch_size,N_MELS), name='harm_input')
    input_layer_P = Input(shape=(patch_size,N_MELS), name='perc_input')
        
    for i in range(n_layers):
        if i == 0:
            x_H = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True, name='tcn_initial_conv_H')(input_layer_H)
        else:
            x_H = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True, name='tcn_H' + str(i))(x_H)

    x_H = Flatten()(x_H)


    for i in range(n_layers):
        if i == 0:
            x_P = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True, name='tcn_initial_conv_P')(input_layer_P)
        else:
            x_P = TCN(list_n_filters[i], kernel_size, nb_stacks, dilations, activation, padding, use_skip_connections, dropout_rate, return_sequences=True, name='tcn_P' + str(i))(x_P)

    x_P = Flatten()(x_P)
    
    x = Concatenate(axis=-1, name='intermediate_fusion_lyr')([x_H, x_P])
    x = BatchNormalization(axis=-1)(x)

    classification_output = Dense(n_classes, activation='softmax', name='3C')(x)

    sp_output, x_sp, mu_output, x_mu, smr_output, x_smr = MTL_modifications(x)
    
    model = Model([input_layer_H, input_layer_P], [sp_output, mu_output, smr_output, classification_output])

    initial_learning_rate = 0.002
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=3*TR_STEPS, decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule, clipnorm=1, momentum=0.9)

    model.compile(
        loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'}, 
        optimizer=optimizer, 
        metrics={'3C':'accuracy'}
        )

    print(model.summary())
    print('MTL modification with intermediate feature fusion for the architecture of Lemaire et. al. Proc. of the 20th ISMIR Conference, Delft, Netherlands, November 4-8, 2019\n')

    return model, initial_learning_rate




def get_Doukhan_MTL_model(PARAMS, n_classes=3):
    '''
    MTL modification of the CNN architecture proposed by Doukhan et al. [1]

        [1] Doukhan, D., Lechapt, E., Evrard, M., & Carrive, J. (2018). Ina’s 
    mirex 2018 music and speech detection system. Music Information Retrieval 
    Evaluation eXchange (MIREX 2018).

    Parameters
    ----------
    PARAMS : dict
        Contains default parameters.
    n_classes : int, optional
        Number of classes. Default is 3.

    Returns
    -------
    model : tensorflow.keras.models.Model
        MTL CNN model.
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

    classification_output = Dense(n_classes, activation='softmax', kernel_initializer=k_init, bias_initializer=b_init, name='3C')(x)

    sp_output, x_sp, mu_output, x_mu, smr_output, x_smr = MTL_modifications(x)
    
    model = Model(input_img, [sp_output, mu_output, smr_output, classification_output])

    learning_rate = 0.0001
    optimizer = optimizers.Adam(lr=learning_rate)    
    model.compile(
        loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'}, 
        optimizer=optimizer, 
        metrics={'3C':'accuracy'}
        )
    
    print(model.summary())
    print('MTL Modification of architecture proposed by Doukhan et al. MIREX 2018 SPEECH MUSIC DETECTION\n')

    return model, learning_rate




def get_Papakostas_MTL_model(PARAMS, n_classes=3):    
    '''
    MTL modification of the CNN architecture proposed by Papakostas et al. [2]    

        [2] Papakostas, M., & Giannakopoulos, T. (2018). Speech-music 
    discrimination using deep visual feature extractors. Expert Systems with 
    Applications, 114, 334-344.

    Parameters
    ----------
    PARAMS : dict
        Contains various parameters.
    n_classes : int, optional
        Number of classes. Default is 3.

    Returns
    -------
    model : tensorflow.keras.models.Model
        MTL CNN model.
    learning_rate : float
        Initial learning rate.

    '''
    input_img = Input(PARAMS['input_shape'][PARAMS['Model']])

    x = Conv2D(96, input_shape=PARAMS['input_shape'][PARAMS['Model']], kernel_size=(5, 5), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(input_img)
    x = Lambda(lambda norm_lyr: LRN(norm_lyr, depth_radius=5, alpha=0.0001, beta=0.75))(x)
    # x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1))(x)
    x = Lambda(lambda norm_lyr: LRN(norm_lyr, depth_radius=5, alpha=0.0001, beta=0.75))(x)
    # x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1), padding='same')(x)
    # x = BatchNormalization(axis=-1)(x)
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

    classification_output = Dense(n_classes, activation='softmax', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(value=0.1), name='3C')(x)

    sp_output, x_sp, mu_output, x_mu, smr_output, x_smr = MTL_modifications(x)
    
    model = Model(input_img, [sp_output, mu_output, smr_output, classification_output])
    
    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=700, decay_rate=0.1)    
    optimizer = optimizers.SGD(learning_rate=lr_schedule)    
    model.compile(
        loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'}, 
        optimizer=optimizer, 
        metrics={'3C':'accuracy'}
        )

    print(model.summary())
    print('MTL modifications of architecture proposed by Papakostas et al. Expert Systems with Applications 2018\n')

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
    kernel_init : array
        Initial weights for the Mel-scale convolutional kernel.

    '''
    kernel_init = w[m[0]:m[1]+1]
    kernel_init = np.repeat(np.array(kernel_init, ndmin=2).T, t_dim, axis=1)
    kernel_init = np.expand_dims(np.expand_dims(kernel_init, axis=2), axis=3)
    kernel_init = np.repeat(kernel_init, 3, axis=3)
    
    return kernel_init



def mel_scale_layer(M, filter_bins, t_dim, n_mels, n_fft, inp_img, name):
    top = filter_bins[0,0]
    bottom = n_fft-filter_bins[0,1]-1
    inp_mel = Cropping2D(cropping=((top, bottom), (0, 0)))(inp_img)
    # print('inp_mel: ', K.int_shape(inp_mel), filter_bins[0,:], top, bottom)
    kernel_width = filter_bins[0,1]-filter_bins[0,0]+1
    kernel_init = get_kernel_initializer(M[0,:], filter_bins[0,:], t_dim)
    melCl = Conv2D(3, kernel_size=(kernel_width,t_dim), strides=(kernel_width,1), padding='same', name=name+'_melCl0', kernel_initializer=Constant(kernel_init), use_bias=False, kernel_regularizer=l2())(inp_mel)

    for mel_i in range(1, n_mels):
        top = filter_bins[mel_i,0]
        bottom = n_fft-filter_bins[mel_i,1]-1
        inp_mel = Cropping2D(cropping=((top, bottom), (0, 0)))(inp_img)
        # print('inp_mel: ', K.int_shape(inp_mel), filter_bins[mel_i,:], top, bottom)
        kernel_width = filter_bins[mel_i,1]-filter_bins[mel_i,0]+1
        kernel_init = get_kernel_initializer(M[mel_i,:], filter_bins[mel_i,:], t_dim)
        melCl_n = Conv2D(3, kernel_size=(kernel_width,t_dim), strides=(kernel_width,1), padding='same', name=name+'_melCl'+str(mel_i), kernel_initializer=Constant(kernel_init), use_bias=False, kernel_regularizer=l2())(inp_mel)
        melCl = Concatenate(axis=1)([melCl, melCl_n])
    ''' ~~~~~~~~~~~~~~~~~~~~ '''
    # melCl = BatchNormalization(axis=-1)(melCl)
    melCl = Activation('tanh')(melCl)
    # melCl = Dropout(0.4)(melCl)
    
    return melCl



def get_Jang_MTL_model(PARAMS, fs=16000, Tw=25, n_mels=120, t_dim=5, n_classes=3):
    '''
    MTL modification of the Mel-scale CNN architecture proposed by 
    Jang et al. [4]
                
        [4] Jang, B. Y., Heo, W. H., Kim, J. H., & Kwon, O. W. (2019). Music 
    detection from broadcast contents using convolutional neural networks with 
    a Mel-scale kernel. EURASIP Journal on Audio, Speech, and Music Processing,
    2019(1), 1-12.
    
    Parameters
    ----------
    fs : int, optional
        Sampling rate. The default is 16000.
    Tw : int, optional
        Short-term frame size in miliseconds. The default is 25.
    n_mels : int, optional
        Number of mel-filters. The default is 120. Optimized with Lemaire et 
        al. model.
    t_dim : int, optional
        Time dimension of the mel-scale kernels. The default is 5.
    n_classes : int, optional
        Number of classes. The default is 3.

    Returns
    -------
    model : tensorflow.keras.models.Model
        Mel-scale CNN MTL model.

    '''
    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    M = librosa.filters.mel(fs, n_fft=n_fft, n_mels=n_mels, norm='slaney')
    # print('M: ', np.shape(M))
    filter_bins = np.zeros((np.shape(M)[0],2))
    for i in range(np.shape(M)[0]):
        bins = np.squeeze(np.where(M[i,:]>0)).flatten()
        if np.isscalar(bins):
            bins = [bins]
        filter_bins[i,0] = bins[0]
        filter_bins[i,1] = bins[-1]
    filter_bins = filter_bins.astype(int)
    n_fft = np.shape(M)[1]
    # print('filter_bins: ', np.shape(filter_bins), n_mels, n_fft)

    inp_img = Input(PARAMS['input_shape'][PARAMS['Model']]) # Input(shape=(514, 68, 1))
    # print('inp_img: ', K.int_shape(inp_img))

    top_H = 0
    bottom_H = 257
    inp_img_H = Cropping2D(cropping=((top_H, bottom_H), (0, 0)), name='harm_crop')(inp_img)
    # print('inp_img_H: ', K.int_shape(inp_img_H))

    top_P = 257
    bottom_P = 0
    inp_img_P = Cropping2D(cropping=((top_P, bottom_P), (0, 0)), name='perc_crop')(inp_img)
    # print('inp_img_P: ', K.int_shape(inp_img_P))

    ''' melCL layers '''
    melCl_H = mel_scale_layer(M, filter_bins, t_dim, n_mels, n_fft, inp_img_H, name='harm')
    melCl_P = mel_scale_layer(M, filter_bins, t_dim, n_mels, n_fft, inp_img_P, name='perc')
    melCl = Concatenate(axis=1)([melCl_H, melCl_P])
    # print('melCl: ', K.int_shape(melCl), K.int_shape(melCl_H), K.int_shape(melCl_P))
    
    x = Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2())(melCl)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)

    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2())(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)

    x = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2())(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
    
    x = Flatten()(x)
    
    # Best result obtained without the fully connected layers
    x = Dense(2048, kernel_regularizer=l2())(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(1024, kernel_regularizer=l2())(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    classification_output = Dense(n_classes, activation='softmax', kernel_regularizer=l2(), name='3C')(x)

    sp_output, x_sp, mu_output, x_mu, smr_output, x_smr = MTL_modifications(x)
    
    model = Model(inp_img, [sp_output, mu_output, smr_output, classification_output])
    
    learning_rate = 0.001
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(
        loss={'S': 'binary_crossentropy', 'M': 'binary_crossentropy', 'R':'mean_squared_error', '3C': 'categorical_crossentropy'}, 
        optimizer=optimizer, 
        metrics={'3C':'accuracy'}
        )
    
    print(model.summary())
    print('MTL modification of Mel-scale CNN architecture proposed by Jang et al.')

    return model, learning_rate
