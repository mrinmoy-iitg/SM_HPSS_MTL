#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:58:39 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import datetime
import numpy as np
import time
import csv
import librosa
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
import lib.misc as misc
from lib.cython_impl.tools import extract_patches as cextract_patches
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import gc
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support





class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()





def start_GPU_session():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 1 , 'CPU': 1}, 
        gpu_options=gpu_options,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        )
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)




def reset_TF_session():
    tf.compat.v1.keras.backend.clear_session()





def getPerformance(PtdLabels, GroundTruths, **kwargs):
    if 'labels' in kwargs.keys():
        labels = kwargs['labels']
    else:
        labels = np.unique(GroundTruths)
    ConfMat = confusion_matrix(y_true=GroundTruths, y_pred=PtdLabels, labels=labels)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true=GroundTruths, y_pred=PtdLabels, beta=1.0, average=None, labels=labels)
    precision = np.round(precision,4)
    recall = np.round(recall,4)
    fscore = np.round(fscore,4)
    accuracy = np.round(np.sum(np.diag(ConfMat))/np.sum(ConfMat), 4)

    return ConfMat, accuracy, precision, recall, fscore




def mode_filtering(X, win_size):
    if win_size%2==0:
        win_size += 1
    X_smooth = X.copy()
    for i in range(int(win_size/2), len(X)-int(win_size/2)):
        win = X[i-int(win_size/2):i+int(win_size/2)]
        uniq_lab, uniq_counts = np.unique(win, return_counts=True)
        X_smooth[i] = uniq_lab[np.argmax(uniq_counts)]    
    return X_smooth




def smooth_labels(Predictions, PtdLabels, win_size, smooth_type='prediction'):
    if smooth_type=='prediction':
        Predictions_smooth = medfilt(Predictions, win_size)
        PtdLabels_smooth = (Predictions_smooth>0.5).astype(int)
        return Predictions_smooth, PtdLabels_smooth
    
    elif smooth_type=='label':
        PtdLabels_smooth = mode_filtering(PtdLabels, win_size)
        return Predictions, PtdLabels_smooth





def plot_segmentation_results(PARAMS, opDirFig, Predictions, PtdLabels_sp, PtdLabels_mu, annot_path, fl, speech_marker, music_marker, win_size):
    PtdLabels_smooth_mu, PtdLabels_smooth_sp = smooth_labels(Predictions, PtdLabels_mu, PtdLabels_sp, win_size, smooth_type='label')
    
    plot_num = 0
    nPlotRows = 4
    nPlotCols = 1
    plt.figure()
    
    plot_num = 1
    plt.subplot(nPlotRows,nPlotCols,plot_num)
    plt.plot(speech_marker, 'm-')
    plt.plot(music_marker*2, 'b-')
    plt.title('Ground Truths')
    plt.legend(['speech', 'music'])

    plot_num = 2
    plt.subplot(nPlotRows,nPlotCols,plot_num)
    plt.plot(PtdLabels_sp, 'm-')
    plt.plot(PtdLabels_mu*2, 'b-')
    plt.title('Classifier labels')
    plt.legend(['speech', 'music'])

    plot_num = 3
    plt.subplot(nPlotRows,nPlotCols,plot_num)
    plt.plot(PtdLabels_smooth_sp, 'm-')
    plt.plot(PtdLabels_smooth_mu*2, 'b-')
    plt.title('Smooth Classifier labels')
    plt.legend(['speech', 'music'])
    
    print('figure plotting')

    # plt.show()
    plt.savefig(opDirFig+fl.split('/')[-1].split('.')[0]+'.jpg', bbox_inches='tight')




def get_annotations(folder, fl, nFrames, opDir): # Designed for OFAI dafx  dataset annotations
    annot_opDir = opDir + '/__annotations/'
    if not os.path.exists(annot_opDir):
        os.makedirs(annot_opDir)
    label_path_mu = folder + '/labels/music/'
    label_path_sp = folder + '/labels/speech/'
    opFile = annot_opDir+'/'+fl.split('.')[0]+'.npz'
    if not os.path.exists(opFile):
        # print('Reading annotations of ', fl)
        max_duration_mu = 0
        annotations_mu = {}
        with open(label_path_mu+'/'+fl+'.csv', newline='\n') as csvfile:
            annotreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_count = 0
            for row in annotreader:
                if row==[]:
                    continue
                if row_count==0:
                    row_count += 1
                    continue
                annotations_mu[row_count] = row
                row_count += 1
                if (float(row[0])+float(row[1]))>max_duration_mu:
                    max_duration_mu = float(row[0])+float(row[1])

        annotations_sp = {}
        max_duration_sp = 0
        with open(label_path_sp+'/'+fl+'.csv', newline='\n') as csvfile:
            annotreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_count = 0
            for row in annotreader:
                if row==[]:
                    continue
                if row_count==0:
                    row_count += 1
                    continue
                annotations_sp[row_count] = row
                row_count += 1
                if (float(row[0])+float(row[1]))>max_duration_sp:
                    max_duration_sp = float(row[0])+float(row[1])
        
        speech_marker = np.zeros(nFrames)
        music_marker = np.zeros(nFrames)
        audio_length = np.max([max_duration_mu, max_duration_sp])
        # print('audio length: ', audio_length)
        
        for row in annotations_mu.keys():
            tmin = float(annotations_mu[row][0])
            dur = float(annotations_mu[row][1])
            label = int(annotations_mu[row][2])
            if dur==0.0:
                continue
            tmax = tmin+dur
            frmStart = np.max([0, int(np.floor((tmin/audio_length)*nFrames))])
            frmEnd = np.min([int(np.ceil((tmax/audio_length)*nFrames)), nFrames-1])
            if label==1:
                music_marker[frmStart:frmEnd] = 1
            # print(row, np.round(tmin,4), np.round(dur,4), np.round(tmax,4), frmStart, frmEnd, label)

        for row in annotations_sp.keys():
            tmin = float(annotations_sp[row][0])
            dur = float(annotations_sp[row][1])
            label = int(annotations_sp[row][2])
            if dur==0.0:
                continue
            tmax = tmin+dur
            frmStart = np.max([0, int(np.floor((tmin/audio_length)*nFrames))])
            frmEnd = np.min([int(np.ceil((tmax/audio_length)*nFrames)), nFrames-1])
            if label==1:
                speech_marker[frmStart:frmEnd] = 1
            # print(row, np.round(tmin,4), np.round(dur,4), np.round(tmax,4), frmStart, frmEnd, label)
        
        np.savez(opFile, annotations_mu=annotations_mu, annotations_sp=annotations_sp, speech_marker=speech_marker, music_marker=music_marker)
    else:
        annotations_mu = np.load(opFile, allow_pickle=True)['annotations_mu']
        annotations_sp = np.load(opFile, allow_pickle=True)['annotations_sp']
        speech_marker = np.load(opFile, allow_pickle=True)['speech_marker']
        music_marker = np.load(opFile, allow_pickle=True)['music_marker']

    return annotations_mu, annotations_sp, music_marker, speech_marker





def get_featuregram(PARAMS, feature_opDir, fName, Spec, n_fft, n_mels, featName, save_feat=True):
    if not os.path.exists(feature_opDir+'/'+fName+'.npy'):
        if featName=='LogMelSpec':
            fv = librosa.feature.melspectrogram(S=Spec, n_mels=n_mels)
            fv = librosa.core.power_to_db(fv**2)
            fv = fv.astype(np.float32)

        elif featName.startswith('LogMelHarm') or featName.startswith('LogMelPerc'):
            H_Spec, P_Spec = librosa.decompose.hpss(S=Spec, kernel_size=(PARAMS['l_harm'][PARAMS['Model']], PARAMS['l_perc'][PARAMS['Model']]))
            fv_H = librosa.feature.melspectrogram(S=H_Spec, n_mels=n_mels)
            fv_H = librosa.core.power_to_db(fv_H**2)
            fv_P = librosa.feature.melspectrogram(S=P_Spec, n_mels=n_mels)
            fv_P = librosa.core.power_to_db(fv_P**2)
            fv = np.append(fv_H, fv_P, axis=0) 
            fv = fv.astype(np.float32)

        if save_feat:
            np.save(feature_opDir+'/'+fName+'.npy', fv)
    else:
        try:
            fv = np.load(feature_opDir+'/'+fName+'.npy', allow_pickle=True)
        except:
            print('Error loading: ', feature_opDir+'/'+fName+'.npy')
            fv = np.load(feature_opDir+'/'+fName+'.npy', allow_pickle=True)
    
    return fv




def get_feature_patches(PARAMS, FV, patch_size, patch_shift, featName):
    # FV should be of the shape (nFeatures, nFrames)
    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)        

    if featName=='LogMelSpec':
        patches = cextract_patches(FV, np.shape(FV), patch_size, patch_shift)
        if not 'Lemaire_et_al' in PARAMS['Model']:
            patches = np.expand_dims(patches, axis=3)
        
    elif featName.startswith('LogMelHarm') or featName.startswith('LogMelPerc'):
        if (featName=='LogMelHarmSpec') or (featName=='LogMelHarmPercSpec'):
            FV_H = FV[:int(np.shape(FV)[0]/2), :] # harmonic
            patches_H = cextract_patches(FV_H, np.shape(FV_H), patch_size, patch_shift)
            if not 'Lemaire_et_al' in PARAMS['Model']:
                patches_H = np.expand_dims(patches_H, axis=3)

        if (featName=='LogMelPercSpec') or (featName=='LogMelHarmPercSpec'):
            FV_P = FV[int(np.shape(FV)[0]/2):, :] # percussive
            patches_P = cextract_patches(FV_P, np.shape(FV_P), patch_size, patch_shift)
            if not 'Lemaire_et_al' in PARAMS['Model']:
                patches_P = np.expand_dims(patches_P, axis=3)
        
        if 'HarmPerc' in featName:
            patches = np.append(patches_H, patches_P, axis=1)
        elif 'Harm' in featName:
            patches = patches_H.copy()
        elif 'Perc' in featName:
            patches = patches_P.copy()
        
        patches = patches.astype(np.float32)

    return patches





def load_data(PARAMS, folder, file_list):
    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    n_mels = PARAMS['n_mels'][PARAMS['Model']]
    featName = PARAMS['featName'][PARAMS['Model']]
    FV = np.empty([], dtype=np.float32)
    labels_mu = np.empty([], dtype=np.int32)
    labels_sp = np.empty([], dtype=np.int32)
    fl_count = 0
    for fName in file_list:
        fl_count += 1
        fName_path = folder + '/features/' + fName + '.npy'
        if not os.path.exists(fName_path):
            continue
        fv = np.load(fName_path, allow_pickle=True)
        fv = get_featuregram(PARAMS, PARAMS['feature_opDir'], fName, fv, n_fft, n_mels, featName, save_feat=True)
        nFrames = np.shape(fv)[1]
        annotations_mu, annotations_sp, music_marker, speech_marker = get_annotations(PARAMS['test_path'], fName, nFrames, PARAMS['opDir'])
        if not 'HarmPerc' in featName:
            fv = fv.T
            fv = StandardScaler(copy=False).fit_transform(fv)
            fv = fv.T
        else:
            nDim = np.shape(fv)[0]
            fv_H = fv[:int(nDim/2),:]
            fv_H = fv_H.T
            fv_H = StandardScaler(copy=False).fit_transform(fv_H)
            fv_H = fv_H.T
            fv_P = fv[int(nDim/2):,:]
            fv_P = fv_P.T
            fv_P = StandardScaler(copy=False).fit_transform(fv_P)
            fv_P = fv_P.T
            fv = np.append(fv_H.astype(np.float32), fv_P.astype(np.float32), axis=0)
        if np.size(FV)<=1:
            FV = fv.astype(np.float32)
            labels_mu = music_marker.astype(np.int32)
            labels_sp = speech_marker.astype(np.int32)
        else:
            FV = np.append(FV, fv.astype(np.float32), axis=1)
            labels_mu = np.append(labels_mu, music_marker.astype(np.int32))
            labels_sp = np.append(labels_sp, speech_marker.astype(np.int32))
        print(fl_count, '/', len(file_list), fName, np.shape(FV), np.shape(labels_mu), np.shape(labels_sp))
    return FV, labels_mu, labels_sp
            



def generator(PARAMS, FV, labels_mu, labels_sp, batchSize):
    batch_count = 0
    batchData_neg_temp = np.empty([], dtype=np.float32)
    batchData_pos_temp = np.empty([], dtype=np.float32)
    balance = [0, 0]
    part_i_neg = 0
    part_j_neg = 0
    part_i_pos = 0
    part_j_pos = 0
    part_size = PARAMS['W_shift']*batchSize*2
    featName = PARAMS['featName'][PARAMS['Model']]

    if PARAMS['signal_type']=='music':
        labels = labels_mu
    elif PARAMS['signal_type']=='speech':
        labels = labels_sp
    neg_idx = np.squeeze(np.where(labels==0))
    pos_idx = np.squeeze(np.where(labels==1))
    FV_neg = FV[:,neg_idx]
    FV_pos = FV[:,pos_idx]

    while 1:
        batchData = np.empty([], dtype=np.float32)
        batchLabel = np.empty([], dtype=np.float32)
        while np.min(balance)<batchSize:
            part_i_neg = part_j_neg
            if part_i_neg>np.shape(FV_neg)[1]:
                part_i_neg = np.random.randint(np.shape(FV_neg)[1])
            part_j_neg = np.min([part_i_neg+part_size, np.shape(FV_neg)[1]])
            if (part_j_neg-part_i_neg)<part_size:
                part_i_neg = 0
            fv_neg = FV[:, part_i_neg:part_j_neg]

            part_i_pos = part_j_pos
            if part_i_pos>np.shape(FV_pos)[1]:
                part_i_pos = np.random.randint(np.shape(FV_pos)[1])
            part_j_pos = np.min([part_i_pos+part_size, np.shape(FV_pos)[1]])
            if (part_j_pos-part_i_pos)<part_size:
                part_i_pos = 0
            fv_pos = FV[:, part_i_pos:part_j_pos]

            if PARAMS['signal_type']=='music':
                W_shift_neg = int(PARAMS['W_shift']/3)
                W_shift_pos = PARAMS['W_shift']
            elif PARAMS['signal_type']=='speech':
                W_shift_neg = PARAMS['W_shift']
                W_shift_pos = int(PARAMS['W_shift']/3)
            
            fv_neg_patches = np.empty([])
            if np.size(neg_idx)>PARAMS['W']:
                fv_neg_patches = get_feature_patches(PARAMS, fv_neg, PARAMS['W'], W_shift_neg, featName)
            fv_pos_patches = np.empty([])
            if np.size(pos_idx)>PARAMS['W']:
                fv_pos_patches = get_feature_patches(PARAMS, fv_pos, PARAMS['W'], W_shift_pos, featName)                
            if np.size(fv_neg_patches)>1:
                if np.size(batchData_neg_temp)<=1:
                    batchData_neg_temp = fv_neg_patches
                else:
                    batchData_neg_temp = np.append(batchData_neg_temp, fv_neg_patches, axis=0)    
                balance[0] += np.shape(fv_neg_patches)[0]
            if np.size(fv_pos_patches)>1:
                if np.size(batchData_pos_temp)<=1:
                    batchData_pos_temp = fv_pos_patches
                else:
                    batchData_pos_temp = np.append(batchData_pos_temp, fv_pos_patches, axis=0)
                balance[1] += np.shape(fv_pos_patches)[0]
        
        batchData = batchData_neg_temp[:batchSize, :, :]
        batchData = np.append(batchData, batchData_pos_temp[:batchSize, :, :], axis=0)
        batchLabel = [0]*batchSize
        batchLabel.extend([1]*batchSize)
        batchLabel = np.array(batchLabel)
        batchData_neg_temp = batchData_neg_temp[batchSize:, :, :]
        batchData_pos_temp = batchData_pos_temp[batchSize:, :, :]
        balance = (np.array(balance)-batchSize).tolist()
        
        if 'Lemaire_et_al' in PARAMS['Model']:
            # TCN input shape=(batch_size, timesteps, ndim)
            batchData = np.transpose(batchData, axes=(0,2,1)) 
            
        ''' Adding Normal (Gaussian) noise for data augmentation '''
        if PARAMS['data_augmentation_with_noise']:
            scale = np.random.choice([5e-3, 1e-3, 5e-4, 1e-4])
            noise = np.random.normal(loc=0.0, scale=scale, size=np.shape(batchData))
            batchData = np.add(batchData, noise)
                            
        batch_count += 1
        # print('Batch ', batch_count, ' shape=', np.shape(batchData), np.shape(batchLabel))
        # print('batchLabel: ', batchLabel)
        
        yield batchData, batchLabel





def transfer_learn_model(PARAMS, model, weightFile, logFile):
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, restore_best_weights=True, min_delta=0.01, patience=5)
    mcp = ModelCheckpoint(weightFile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    csv_logger = CSVLogger(logFile)

    trainingTimeTaken = 0
    start = time.process_time()

    SPE = PARAMS['TR_STEPS']
    SPE_val = PARAMS['V_STEPS']
    print('SPE: ', SPE, SPE_val)

    train_data, tr_labels_mu, tr_labels_sp = load_data(PARAMS, PARAMS['test_path'], PARAMS['train_files'])
    print('train_data: ', np.shape(train_data))
    val_data, v_labels_mu, v_labels_sp = load_data(PARAMS, PARAMS['test_path'], PARAMS['val_files'])
    print('val_data: ', np.shape(val_data))

    # Train the model
    History = model.fit(
            generator(PARAMS, train_data, tr_labels_mu, tr_labels_sp, PARAMS['batch_size']),
            steps_per_epoch = SPE,
            validation_data = generator(PARAMS, val_data, v_labels_mu, v_labels_sp, PARAMS['batch_size']),
            validation_steps = SPE_val,
            epochs=PARAMS['epochs'], 
            verbose=1,
            callbacks=[csv_logger, es, mcp, ClearMemory()],
            )

    trainingTimeTaken = time.process_time() - start
    print('Time taken for model training: ',trainingTimeTaken)

    return model, trainingTimeTaken, History
    




def load_model(PARAMS, modelName, updated_modelName):
    modelName = '@'.join(modelName.split('.')[:-1]) + '.' + modelName.split('.')[-1]
    weightFile = modelName.split('.')[0] + '.h5'
    architechtureFile = modelName.split('.')[0] + '.json'
    paramFile = modelName.split('.')[0] + '_params.npz'
    logFile = modelName.split('.')[0] + '_log.csv'

    modelName = '.'.join(modelName.split('@'))
    weightFile = '.'.join(weightFile.split('@'))
    architechtureFile = '.'.join(architechtureFile.split('@'))
    paramFile = '.'.join(paramFile.split('@'))
    logFile = '.'.join(logFile.split('@'))
    print('paramFile: ', paramFile)
    print('weightFile: ', weightFile)
    
    epochs = np.load(paramFile)['epochs']
    batch_size = np.load(paramFile)['batch_size']
    learning_rate = np.load(paramFile)['lr']
    trainingTimeTaken = np.load(paramFile)['trainingTimeTaken']
    with open(architechtureFile, 'r') as f: # Model reconstruction from JSON file
        trained_model = model_from_json(f.read())
    trained_model.load_weights(weightFile) # Load weights into the new model
    weightFile = None
    architechtureFile = None
    paramFile = None
    logFile = None
    
    updated_modelName = '@'.join(updated_modelName.split('.')[:-1]) + '.' + updated_modelName.split('.')[-1]
    updated_weightFile = updated_modelName.split('.')[0] + '.h5'
    updated_architechtureFile = updated_modelName.split('.')[0] + '.json'
    updated_paramFile = updated_modelName.split('.')[0] + '_params.npz'
    updated_logFile = updated_modelName.split('.')[0] + '_log.csv'

    updated_modelName = '.'.join(updated_modelName.split('@'))
    updated_weightFile = '.'.join(updated_weightFile.split('@'))
    updated_architechtureFile = '.'.join(updated_architechtureFile.split('@'))
    updated_paramFile = '.'.join(updated_paramFile.split('@'))
    updated_logFile = '.'.join(updated_logFile.split('@'))

    if PARAMS['signal_type']=='music':
        mu_output = trained_model.get_layer('M').output
        model = Model(trained_model.input, mu_output)
    elif PARAMS['signal_type']=='speech':
        sp_output = trained_model.get_layer('S').output
        model = Model(trained_model.input, sp_output)

    learning_rate = 0.002
    optimizer = optimizers.Nadam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='accuracy')
    print(model.summary())
    print('Loaded trained Lemaire et. al. MTL model for ', PARAMS['signal_type'], ' detection on DAFx2012 dataset\n')
    print('input shape: ', K.int_shape(model.input))
    print('output shape: ', K.int_shape(model.output))

    if PARAMS['use_updated_model']:    
        if not os.path.exists(updated_paramFile):
            if os.path.exists(updated_weightFile):
                model.load_weights(updated_weightFile) # Load weights into the new model
                epoch_count = 0
                if os.path.exists(updated_logFile):
                    with open(updated_logFile, 'r', encoding='utf8') as fid:
                        for line in fid:
                            epoch_count += 1
                PARAMS['epochs'] -= epoch_count
                learning_rate = 0.002
                optimizer = optimizers.Nadam(learning_rate=learning_rate)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='accuracy')
            else:
                misc.print_model_summary(PARAMS['opDir'] + '/model_summary.txt', model)            
            if PARAMS['epochs']>0:
                model, trainingTimeTaken, History = transfer_learn_model(PARAMS, model, updated_weightFile, updated_logFile)
            if PARAMS['save_flag']:
                # model.save_weights(updated_weightFile) # Save the weights
                with open(updated_architechtureFile, 'w') as f: # Save the model architecture
                    f.write(model.to_json())
                np.savez(updated_paramFile, epochs=PARAMS['epochs'], batch_size=PARAMS['batch_size'], lr=learning_rate, trainingTimeTaken=trainingTimeTaken)
            print('Updated model trained.')
            
        else:
            epochs = np.load(updated_paramFile)['epochs']
            batch_size = np.load(updated_paramFile)['batch_size']
            learning_rate = np.load(updated_paramFile)['lr']
            trainingTimeTaken = np.load(updated_paramFile)['trainingTimeTaken']
            optimizer = optimizers.Adam(lr=learning_rate)
            if not os.path.exists(updated_architechtureFile):
                with open(updated_architechtureFile, 'w') as f: # Save the model architecture
                    f.write(model.to_json())
            else:
                with open(updated_architechtureFile, 'r') as f: # Model reconstruction from JSON file
                    model = model_from_json(f.read())
            model.load_weights(updated_weightFile) # Load weights into the new model
    
            optimizer = optimizers.Nadam(learning_rate=0.002)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            print('Updated model exists! Loaded. Training time required=',trainingTimeTaken)
    else:
        model = trained_model
        
    Train_Params = {
            'model': model,
            'trainingTimeTaken': trainingTimeTaken,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'paramFile': updated_paramFile,
            'architechtureFile': updated_architechtureFile,
            'weightFile': updated_weightFile,
            }
    
    return Train_Params





def patch_probability_generator(PARAMS, fl, Train_Params):
    startTime = time.process_time()
    labels_sp = []
    labels_mu = []
    pred_opDir = PARAMS['opDir'] + '/__Frame_Predictions_CNN/'
    if not os.path.exists(pred_opDir):
        os.makedirs(pred_opDir)
    result_fName = fl + '_fold' + str(PARAMS['fold']) +  '_result'
    
    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    n_mels = PARAMS['n_mels'][PARAMS['Model']]
    featName = PARAMS['featName'][PARAMS['Model']]
    
    if not os.path.exists(pred_opDir+result_fName+'.pkl'):
        fName_path = PARAMS['test_path'] + '/features/' + fl + '.npy'
        if not os.path.exists(fName_path):
            return {}
        fv = np.load(fName_path, allow_pickle=True)
        fv = get_featuregram(PARAMS, PARAMS['feature_opDir'], fl, fv, n_fft, n_mels, featName, save_feat=True)
        if not 'HarmPerc' in featName:
            fv = fv.T
            fv = StandardScaler(copy=False).fit_transform(fv)
            fv = fv.T
        else:
            nDim = np.shape(fv)[0]
            fv_H = fv[:int(nDim/2),:]
            fv_H = fv_H.T
            fv_H = StandardScaler(copy=False).fit_transform(fv_H)
            fv_H = fv_H.T
            fv_P = fv[int(nDim/2):,:]
            fv_P = fv_P.T
            fv_P = StandardScaler(copy=False).fit_transform(fv_P)
            fv_P = fv_P.T
            fv = np.append(fv_H.astype(np.float32), fv_P.astype(np.float32), axis=0)

        nFrames = np.shape(fv)[1]
        annotations_mu, annotations_sp, music_marker, speech_marker = get_annotations(PARAMS['test_path'], fl, nFrames, PARAMS['opDir'])

        pred = np.empty([])
        pred_lab = np.empty([])        
        batch_size = 10000
        labels_mu = []
        labels_sp = []
        # for batchStart in range(0, np.shape(fv_patches)[0], batch_size):
        for batchStart in range(0, np.shape(fv)[1], batch_size):
            # batchEnd = np.min([batchStart+batch_size, np.shape(fv_patches)[0]])
            batchEnd = np.min([batchStart+batch_size, np.shape(fv)[1]])
            # fv_patches_temp = fv_patches[batchStart:batchEnd,:]
            fv_temp = fv[:,batchStart:batchEnd]
            music_marker_temp =  music_marker[batchStart:batchEnd]
            speech_marker_temp =  speech_marker[batchStart:batchEnd]
            print('\tBatch: (', batchStart, batchEnd, ') ', np.shape(fv_temp), ' mu=', np.sum(music_marker_temp), ' sp=', np.sum(speech_marker_temp), end=' ', flush=True)

            fv_patches_temp = get_feature_patches(PARAMS, fv_temp, PARAMS['W'], PARAMS['W_shift_test'], featName)
    
            labels_mu_patches = cextract_patches(np.array(music_marker_temp, ndmin=2), np.shape(np.array(music_marker_temp, ndmin=2)), PARAMS['W'], PARAMS['W_shift_test']).astype(int)
            labels_mu_temp = ((np.sum(np.squeeze(labels_mu_patches, axis=1), axis=1)/np.shape(labels_mu_patches)[2])>0.5).astype(int)
            
            labels_sp_patches = cextract_patches(np.array(speech_marker_temp, ndmin=2), np.shape(np.array(speech_marker_temp, ndmin=2)), PARAMS['W'], PARAMS['W_shift_test']).astype(int)
            labels_sp_temp = ((np.sum(np.squeeze(labels_sp_patches, axis=1), axis=1)/np.shape(labels_sp_patches)[2])>0.5).astype(int)
    
            if 'Lemaire_et_al' in PARAMS['Model']:
                # TCN input shape=(batch_size, timesteps, ndim)
                fv_patches_temp = np.transpose(fv_patches_temp, axes=(0,2,1))

            if PARAMS['signal_type']=='music':
                pred_temp = Train_Params['model'].predict(x=fv_patches_temp)
                CM, acc, P, R, F1 = getPerformance(np.array((pred_temp>0.5).astype(int)), labels_mu_temp)
            elif PARAMS['signal_type']=='speech':
                pred_temp = Train_Params['model'].predict(x=fv_patches_temp)
                CM, acc, P, R, F1 = getPerformance(np.array((pred_temp>0.5).astype(int)), labels_sp_temp)

            pred_lab_temp = np.array(pred_temp>0.5).astype(int)
                
            if np.size(pred)<=1:
                pred = pred_temp
                pred_lab = pred_lab_temp
            else:
                pred = np.append(pred, pred_temp)
                pred_lab = np.append(pred_lab, pred_lab_temp)
            labels_mu.extend(labels_mu_temp)
            labels_sp.extend(labels_sp_temp)
            print(np.shape(fv_patches_temp), np.shape(pred_temp), np.shape(pred), ' acc=', acc, F1)
        
        if PARAMS['signal_type']=='music':
            ConfMat, precision, recall, fscore = misc.getPerformance(pred_lab, labels_mu, labels=[0,1])
            acc = np.round(np.sum(np.diag(ConfMat))/np.sum(ConfMat),4)
            print('Perf mu: ', acc, precision, recall, fscore)
        elif PARAMS['signal_type']=='speech':
            ConfMat, precision, recall, fscore = misc.getPerformance(pred_lab, labels_sp, labels=[0,1])
            acc = np.round(np.sum(np.diag(ConfMat))/np.sum(ConfMat),4)
            print('Perf sp: ', acc, precision, recall, fscore)
        print('\n\n\n')
        
        probability_genTime = time.process_time() - startTime
        result = {
            'pred': pred,
            'pred_lab':pred_lab,
            'labels_sp':labels_sp,
            'labels_mu':labels_mu,
            'probability_genTime': probability_genTime,
            'ConfMat': ConfMat,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'accuracy': acc,
            }
        misc.save_obj(result, pred_opDir, result_fName)
        print('Test predictions saved!!!')
    else:
        result = misc.load_obj(pred_opDir, result_fName)
        
    return result





def performance_dump(PARAMS, PtdLabels, GroundTruths, labels, info='', fName_suffix=''):
    ConfMat, precision, recall, fscore = misc.getPerformance(PtdLabels, GroundTruths, labels)
    accuracy = np.round(np.sum(np.diag(ConfMat))/np.sum(ConfMat),4)
    print('Total data performance: ', fscore)
    print(ConfMat)
    
    if len(labels)==2:
        classnames = ['neg', 'pos']
    else:
        classnames = ['mu', 'sp', 'spmu']

    res_dict = {}
    res_dict['0'] = 'feature_name:'+PARAMS['featName'][PARAMS['Model']]
    res_dict['1'] = 'model:'+PARAMS['Model']
    ln = 2
    if not info=='':
        res_dict[str(ln)] = info
        ln += 1
    res_dict[str(ln)] = 'loss:--'
    ln += 1
    res_dict[str(ln)] = 'accuracy:' + str(accuracy)
    ln += 1
    res_dict[str(ln)] = 'Prec_' + classnames[0] + ':' + str(precision[0])
    ln += 1
    res_dict[str(ln)] = 'Rec_' + classnames[0] + ':' + str(recall[0])
    ln += 1
    res_dict[str(ln)] = 'F1_' + classnames[0] + ':' + str(fscore[0])
    ln += 1
    res_dict[str(ln)] = 'Prec_' + classnames[1] + ':' + str(precision[1])
    ln += 1
    res_dict[str(ln)] = 'Rec_' + classnames[1] + ':' + str(recall[1])
    ln += 1
    res_dict[str(ln)] = 'F1_' + classnames[1] + ':' + str(fscore[1])
    if len(labels)==3:
        ln += 1
        res_dict[str(ln)] = 'Prec_' + classnames[2] + ':' + str(precision[2])
        ln += 1
        res_dict[str(ln)] = 'Rec_' + classnames[2] + ':' + str(recall[2])
        ln += 1
        res_dict[str(ln)] = 'F1_' + classnames[2] + ':' + str(fscore[2])
    ln += 1
    res_dict[str(ln)] = 'F1_avg:' + str(np.round(np.mean(fscore),4))
    misc.print_results(PARAMS, fName_suffix, res_dict)




def __init__():
    patch_size = 99 # 68 
    patch_shift = 34 # 34
    opt_n_mels = 120
    opt_l_harm = 21
    opt_l_perc = 11
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'feature_folder': './features/', 
            # 'feature_folder': '/home1/PhD/mrinmoy.bhattacharjee/features/', # EEE GPU
            # 'model_folder': './results/musan/Baseline/Lemaire_et_al/LogMelSpec_3classes_680ms_120mels/', 
            # 'model_folder': './results/musan/Proposed_Work/Lemaire_et_al_MTL/4_outputs/LogMelHarmPercSpec_3classes_680ms_120mels_21l_harm_11l_perc/', # 695ms model
            'model_folder': './results/musan/Proposed_Work/Lemaire_et_al_MTL/Patch_Size_Experiment/LogMelHarmPercSpec_3classes_1000ms/', # 1000ms model
            # 'test_path': '/home/phd/mrinmoy.bhattacharjee/data/dafx2012/', # EEE GPU
            'test_path': '/scratch/mbhattacharjee/data/dafx2012/', # PARAM-ISHAN
            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'use_GPU': True,
            'GPU_session':None,
            'classes': {0:'music', 1:'speech'}, # {0:'music', 1:'speech', 2:'speech_music'},
            'data_augmentation_with_noise': True,
            'signal_type': 'music', # 'music', 'speech'
            'Model': 'Lemaire_et_al_MTL',
            'featName': {'Lemaire_et_al_MTL':'LogMelHarmPercSpec'},
            'n_fft': {'Lemaire_et_al':400,'Lemaire_et_al_MTL':400},
            'n_mels': {'Lemaire_et_al_MTL':opt_n_mels},
            'l_harm': {'Lemaire_et_al_MTL':opt_l_harm},
            'l_perc': {'Lemaire_et_al_MTL':opt_l_perc},
            'input_shape': {'Lemaire_et_al_MTL':(patch_size,opt_n_mels,1)},
            'W':patch_size,
            'W_shift':patch_shift,
            'W_shift_test': 1,
            'Tw': 25,
            'Ts': 10,
            'epochs': 50,
            'batch_size': 16,
            'train_files': ['527-RSR3-2012_01_30_11_30_01', '160-RSR3-2012_01_26_15_45_02', '454-RSR3-2012_01_29_17_15_01', '418-RSR3-2012_01_29_08_15_02', '581-RSR3-2012_01_31_01_00_01', '661-RSI2-2012_01_31_21_00_01', '383-RSI2-2012_01_28_23_30_01', '31-RSI2-2012_01_25_07_30_02', '537-RSI2-2012_01_30_14_00_01', '434-RSI2-2012_01_29_12_15_02', '669-Chablais-2012_01_31_23_00_01', '522-Chablais-2012_01_30_10_15_01', '236-Chablais-2012_01_26_10_45_02', '110-Chablais-2012_01_26_03_15_02', '343-Chablais-2012_01_28_13_30_02', '371-Rumantsch-2012_01_28_20_30_01', '194-Rumantsch-2012_02_01_00_15_01', '610-Rumantsch-2012_01_31_08_15_01', '354-Rumantsch-2012_01_28_16_15_02', '89-Rumantsch-2012_01_25_22_00_01', '56-drsvirus-2012_01_25_13_45_01', '386-drsvirus-2012_01_29_00_15_01', '204-drsvirus-2012_02_01_02_45_01', '554-drsvirus-2012_01_30_18_15_01', '519-drsvirus-2012_01_30_09_30_02', '280-Central-2012_01_27_21_45_01', '165-Central-2012_01_26_17_00_01', '466-Central-2012_01_29_20_15_01', '615-Central-2012_01_31_09_30_01', '656-Central-2012_01_31_19_45_01'],
            'val_files': ['653-RSR3-2012_01_31_19_00_02', '573-RSR3-2012_01_30_23_00_01', '221-RSI2-2012_02_01_07_00_02', '361-RSI2-2012_01_28_18_00_02', '371-Chablais-2012_01_28_20_30_01', '70-Chablais-2012_01_25_17_15_04', '141-Rumantsch-2012_01_26_11_00_02', '569-Rumantsch-2012_01_30_22_00_01', '256-drsvirus-2012_01_27_15_45_03', '659-drsvirus-2012_01_31_20_30_01', '117-Central-2012_01_26_05_00_01', '58-Central-2012_01_25_14_15_01'],
            'test_files': {
                'austrian':['LIF-h3', 'LIF-h1', 'LIF-h2', 'OE3-h1', 'OE3-h3', 'OE3-h2', 'FM4-h3', 'FM4-h2', 'FM4-h1', 'OE1-h3', 'OE1-h1', 'OE1-h2'],
                'swiss': ['332-RSR3-2012_02_01_10_45_01', '54-RSR3-2012_01_25_13_15_02', '317-RSR3-2012_01_28_07_00_01', '293-RSI2-2012_01_28_01_00_03', '451-RSI2-2012_01_29_16_30_01', '136-RSI2-2012_01_26_09_45_03', '85-Chablais-2012_01_25_21_00_01', '146-Chablais-2012_01_26_12_15_06', '386-Chablais-2012_01_29_00_15_01', '173-Rumantsch-2012_02_01_19_00_01', '244-Rumantsch-2012_01_27_12_45_01', '402-Rumantsch-2012_01_29_04_15_01', '411-drsvirus-2012_01_29_06_30_02', '281-drsvirus-2012_01_27_22_00_01', '610-drsvirus-2012_01_31_08_18_01', '487-Central-2012_01_30_01_31_01', '528-Central-2012_01_30_11_45_01', '219-Central-2012_01_26_06_30_01'],
                },
            'smoothing_win_size': 501,
            'dataset':'',
            'plot_fig': False,
            'use_updated_model': True,
            }

    n_classes = 2
    DT_SZ_tr = 54024700
    DT_SZ_vl = 21666140
    DT_SZ_ts = 75781960
    shft = PARAMS['W_shift']*PARAMS['Ts'] # Interval shift in milisecs
    PARAMS['TR_STEPS'] = int(np.floor(DT_SZ_tr/shft)/(n_classes*PARAMS['batch_size']))
    PARAMS['V_STEPS'] = int(np.floor(DT_SZ_vl/shft)/(n_classes*PARAMS['batch_size']))
    PARAMS['TS_STEPS'] = int(np.floor(DT_SZ_ts/shft)/(n_classes*PARAMS['batch_size']))
    print('TR_STEPS: %d, \tV_STEPS: %d,  \tTS_STEPS: %d\n'%(PARAMS['TR_STEPS'], PARAMS['V_STEPS'], PARAMS['TS_STEPS']))

    return PARAMS



if __name__ == '__main__':
    PARAMS = __init__()
    
    if 'HarmPerc' in PARAMS['featName'][PARAMS['Model']]:
        inp_shape = PARAMS['input_shape'][PARAMS['Model']]
        if 'Lemaire_et_al' in PARAMS['Model']:
            PARAMS['input_shape'][PARAMS['Model']] = (inp_shape[0],2*inp_shape[1],1)
        else:
            PARAMS['input_shape'][PARAMS['Model']] = (2*inp_shape[0],inp_shape[1],1)
        
    PARAMS['feature_opDir'] = PARAMS['feature_folder'] + '/OFAI_dafx/' + PARAMS['Model'] + '/' + PARAMS['featName'][PARAMS['Model']] + '/'
    if not os.path.exists(PARAMS['feature_opDir']):
        os.makedirs(PARAMS['feature_opDir'])
    
    PARAMS['opDir'] = './results/OFAI_dafx/Segmentation_results/' + PARAMS['Model'] + '/' + PARAMS['featName'][PARAMS['Model']] + '_' + PARAMS['signal_type'] + '_' + PARAMS['today'] + '/' 

    if not os.path.exists(PARAMS['opDir']):
        os.makedirs(PARAMS['opDir'])
                
    misc.print_configuration(PARAMS)

    for PARAMS['fold'] in range(0, 1): # range(PARAMS['CV_folds']):
        if PARAMS['use_GPU']:
            start_GPU_session()

        print('\nfoldNum: ', PARAMS['fold'])
        PARAMS['modelName'] = PARAMS['model_folder'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
        PARAMS['updated_modelName'] = PARAMS['opDir'] + '/fold' + str(PARAMS['fold']) + '_model.xyz'
        Train_Params = load_model(PARAMS, PARAMS['modelName'], PARAMS['updated_modelName'])
        
        for test_file_type in PARAMS['test_files'].keys():
            test_files = PARAMS['test_files'][test_file_type]
            
            Metrics = {}
            All_Predictions = []
            All_Predictions_smooth = []
            All_PtdLabels = []
            All_PtdLabels_smooth = []
            All_GroundTruths = []
            fl_count = 0
            for fl in test_files:
                fl_count += 1
                print(fl_count, '/', len(test_files), test_file_type, fl)
                Predictions_Dict = patch_probability_generator(PARAMS, fl, Train_Params)        
                Metrics[fl] = Predictions_Dict
                All_Predictions.extend(Predictions_Dict['pred'].flatten())
                All_PtdLabels.extend(Predictions_Dict['pred_lab'])
                if PARAMS['signal_type']=='music':
                    All_GroundTruths.extend(Predictions_Dict['labels_mu'])
                elif PARAMS['signal_type']=='speech':
                    All_GroundTruths.extend(Predictions_Dict['labels_sp'])
    
                pred_smooth, PtdLabel_smooth = smooth_labels(Predictions_Dict['pred'], Predictions_Dict['pred_lab'], win_size=PARAMS['smoothing_win_size'], smooth_type='prediction')
                All_Predictions_smooth.extend(pred_smooth)
                All_PtdLabels_smooth.extend(PtdLabel_smooth)
                
            performance_dump(PARAMS, All_PtdLabels, All_GroundTruths, labels=[0,1], fName_suffix='segment_level_'+test_file_type)
            performance_dump(PARAMS, All_PtdLabels_smooth, All_GroundTruths, labels=[0,1], info='smoothing_win_size:'+str(PARAMS['smoothing_win_size']), fName_suffix='segment_level_smooth_'+test_file_type)
    
            for fl in test_files:
                if not fl in Metrics.keys():
                    continue
                print(fl, ' fscore: ', Metrics[fl]['fscore'])
       
                ''' Plot segmentation results '''
                if PARAMS['plot_fig']:
                    opDirFig = PARAMS['opDir'] + '/__figures/fold' + str(PARAMS['fold']) + '/'
                    if not os.path.exists(opDirFig):
                        os.makedirs(opDirFig)
       
                    plot_segmentation_results(PARAMS, opDirFig, Metrics[fl]['pred'], Metrics[fl]['pred_lab_sp'], Metrics[fl]['pred_lab_mu'], PARAMS['annot_path'], fl, Metrics[fl]['labels_sp'], Metrics[fl]['labels_mu'], win_size=PARAMS['smoothing_win_size'])
                    
            if PARAMS['use_GPU']:
                reset_TF_session()
