#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:28:27 2019

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import PatchExtractor
import os





def removeSilence(Xin, fs, Tw, Ts, alpha=0.05, beta=0.1):   
    frameSize = int((Tw*fs)/1000) # Frame size in number of samples
    frameShift = int((Ts*fs)/1000) # Frame shift in number of samples
    
    Rmse = librosa.feature.rms(y=Xin, frame_length=frameSize, hop_length=frameShift)
    energy = Rmse[0,:] #pow(Rmse,2)
    energyThresh = alpha*np.max(energy) # TEST WITH MEAN

    frame_silMarker = energy
    frame_silMarker[frame_silMarker < energyThresh] = 0
    frame_silMarker[frame_silMarker >= energyThresh] = 1
    silences = np.empty([])
    totalSilDuration = 0

    # Suppressing spurious noises -----------------------------------------        
    winSz = 20 # Giving a window size of almost 105ms for 10ms frame and 5ms shift
    i = winSz
    while i < len(frame_silMarker)-winSz:
        if np.sum(frame_silMarker[i-int(winSz/2):i+int(winSz/2)]==1) <= np.ceil(winSz*0.3):
            frame_silMarker[i] = 0
        i = i + 1
    # ---------------------------------------------------------------------
    
    sample_silMarker = np.ones(len(Xin))
    i=0
    while i<len(frame_silMarker):
        while frame_silMarker[i]==1:
            if i == len(frame_silMarker)-1:
                break
            i = i + 1
        j = i
        while frame_silMarker[j]==0:
            if j == len(frame_silMarker)-1:
                break
            j = j + 1
        k = np.max([frameShift*(i-1)+frameSize, 1])
        l = np.min([frameShift*(j-1)+frameSize,len(Xin)]);
        
        # Only silence segments of durations greater than given beta
        # (e.g. 100ms) are removed
        if (l-k)/fs > beta:
            sample_silMarker[k:l] = 0
            if np.size(silences)<=1:
                silences = np.array([k,l], ndmin=2)
            else:
                silences = np.append(silences, np.array([k,l], ndmin=2),0)
            totalSilDuration += (l-k)/fs
        i = j + 1
    
    if np.size(silences)>1:
        Xin_silrem = np.empty([]) #Xin
        for i in range(np.shape(silences)[0]):
            if i==0:
                Xin_silrem = Xin[:silences[i,0]]
            else:
                Xin_silrem = np.append(Xin_silrem, Xin[silences[i-1,1]:silences[i,0]])
    else:
        Xin_silrem = Xin
        
    return Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration




def normalize_signal(Xin):
    Xin = Xin - np.mean(Xin)
    Xin = Xin / np.max(np.abs(Xin))
    return Xin




def load_and_preprocess_signal(fName, Tw, Ts, silThresh):
    Xin, fs = librosa.core.load(fName, mono=True, sr=16000)
    Xin = librosa.effects.preemphasis(Xin)
    if silThresh>0:
        Xin = normalize_signal(Xin)
        Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration = removeSilence(Xin, fs, Tw, Ts, silThresh)
        if np.size(Xin_silrem)<=1:
            Xin = Xin_silrem
        Xin = Xin_silrem.copy()

        if len(Xin)/fs < 0.1:
            while len(Xin)/fs<0.1:
                Xin = np.append(Xin, Xin)
    Xin_norm = normalize_signal(Xin)
    
    return Xin_norm, fs



def extract_patches(FV, patch_size, patch_shift):
    patches = np.empty([])
    padding = np.ones((np.shape(FV)[0], int(np.floor(patch_size/2))))*1e-5
    FV = np.append(padding, FV, axis=1)
    FV = np.append(FV, padding, axis=1)
    for frm in range(int(np.ceil(patch_size/2)), np.shape(FV)[1], patch_shift):
        fv_temp = np.expand_dims(FV[:,frm-int(np.floor(patch_size/2)):frm+int(np.floor(patch_size/2))], axis=0)
        if np.size(patches)<=1:
            patches = fv_temp
        else:
            patches = np.append(patches, fv_temp, axis=0)
        print('patches: ', np.shape(patches))
    return patches




def get_feature_patches(PARAMS, FV, patch_size, patch_shift, featName):
    # FV should be of the shape (nFeatures, nFrames)
    if np.shape(FV)[1]<patch_size:
        FV1 = FV.copy()
        while np.shape(FV)[1]<=patch_size:
            FV = np.append(FV, FV1, axis=1)        
    
    if featName=='Melspectrogram':
        FV = StandardScaler(copy=False).fit_transform(FV)
        if PARAMS['task']=='Classification':
            patches = np.empty([])    
            numPatches = int(np.ceil(np.shape(FV)[1]/patch_shift))
            patches = PatchExtractor(patch_size=(np.shape(FV)[0], patch_size), max_patches=numPatches).transform(np.expand_dims(FV, axis=0))
        elif PARAMS['task']=='Segmentation':
            patches = extract_patches(FV, patch_size, patch_shift)        
        patches = np.expand_dims(patches, axis=3)
        
    elif featName=='MelHPSS':
        if (PARAMS['HPSS_type']=='Harmonic') or (PARAMS['HPSS_type']=='Both'):
            FV_H = FV[:int(np.shape(FV)[0]/2), :] # harmonic
            FV_H = StandardScaler(copy=False).fit_transform(FV_H)
            if PARAMS['task']=='Classification':
                patches_H = np.empty([])
                numPatches = int(np.ceil(np.shape(FV_H)[1]/patch_shift))
                patches_H = PatchExtractor(patch_size=(np.shape(FV_H)[0], patch_size), max_patches=numPatches).transform(np.expand_dims(FV_H, axis=0))
            elif PARAMS['task']=='Segmentation':
                patches_H = extract_patches(FV_H, patch_size, patch_shift)
            patches_H = np.expand_dims(patches_H, axis=3)

        if (PARAMS['HPSS_type']=='Percussive') or (PARAMS['HPSS_type']=='Both'):
            FV_P = FV[int(np.shape(FV)[0]/2):, :] # percussive
            FV_P = StandardScaler(copy=False).fit_transform(FV_P)
            if PARAMS['task']=='Classification':
                patches_P = np.empty([])
                numPatches = int(np.ceil(np.shape(FV_P)[1]/patch_shift))
                patches_P = PatchExtractor(patch_size=(np.shape(FV_P)[0], patch_size), max_patches=numPatches).transform(np.expand_dims(FV_P, axis=0))
            elif PARAMS['task']=='Segmentation':
                patches_P = extract_patches(FV_P, patch_size, patch_shift)
            patches_P = np.expand_dims(patches_P, axis=3)
        
        if PARAMS['HPSS_type']=='Both':
            patches = np.append(patches_H, patches_P, axis=3)
        elif PARAMS['HPSS_type']=='Harmonic':
            patches = patches_H.copy()
        elif PARAMS['HPSS_type']=='Percussive':
            patches = patches_P.copy()

    return patches




def mix_signals(Xin1, Xin2, target_dB):
    sig1_len = len(Xin1)
    sig2_len = len(Xin2)
    Xin2_temp = Xin2.copy()
    while sig2_len<sig1_len:
        Xin2_temp = np.append(Xin2_temp, Xin2)
        sig2_len = len(Xin2_temp)
    Xin2 = Xin2_temp.copy()
    common_len = np.min([sig1_len, sig2_len])
    Xin1 = Xin1[:common_len]
    Xin2 = Xin2[:common_len]
    
    sig1_energy = np.sum(np.power(Xin1,2))/len(Xin1)
    sig2_energy = np.sum(np.power(Xin2,2))/len(Xin2)
    
    req_sig2_energy = sig1_energy/np.power(10,(target_dB/10))
    sig2_mult_fact = np.sqrt(req_sig2_energy/sig2_energy)
    Xin2_scaled = sig2_mult_fact*Xin2
    
    Xin_mix = Xin1 + Xin2_scaled
    Xin_mix = normalize_signal(Xin_mix)
    
    return Xin_mix





def get_featuregram(PARAMS, classname, feature_opDir, fName_path1, fName_path2, target_dB):
    if (fName_path1!='') and (fName_path2!=''): # speech_music
        fName = fName_path1.split('/')[-1].split('.')[0]+'_'+str(target_dB)+'dB'
    elif fName_path1!='': # speech
        fName = fName_path1.split('/')[-1].split('.')[0]
    elif fName_path2!='': # music
        fName = fName_path2.split('/')[-1].split('.')[0]

    if not os.path.exists(feature_opDir+'/'+classname+'/'+fName+'.npy'):
        if classname=='speech_music': #(fName_path1!='') and (fName_path2!=''):
            Xin_sp, fs = load_and_preprocess_signal(fName_path1, PARAMS['Tw'], PARAMS['Ts'], PARAMS['silThresh'])
            Xin_mu, fs = load_and_preprocess_signal(fName_path2, PARAMS['Tw'], PARAMS['Ts'], PARAMS['silThresh'])
            Xin = mix_signals(Xin_sp, Xin_mu, target_dB)
    
        elif classname=='speech': # fName_path1!='':
            Xin, fs = load_and_preprocess_signal(fName_path1, PARAMS['Tw'], PARAMS['Ts'], PARAMS['silThresh'])
    
        elif classname=='music': # fName_path2!='':
            Xin, fs = load_and_preprocess_signal(fName_path2, PARAMS['Tw'], PARAMS['Ts'], PARAMS['silThresh'])

        elif classname=='muspeak': # fName_path2!='':
            Xin, fs = load_and_preprocess_signal(fName_path1, PARAMS['Tw'], PARAMS['Ts'], PARAMS['silThresh'])

        if PARAMS['featName']=='Melspectrogram':
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            fv = librosa.feature.melspectrogram(y=Xin, sr=fs, n_fft=frameSize, hop_length=frameShift, center=False, n_mels=PARAMS['n_mels'])
            fv = fv.astype(np.float32)

        elif PARAMS['featName']=='MelHPSS':
            frameSize = int(PARAMS['Tw']*fs/1000)
            frameShift = int(PARAMS['Ts']*fs/1000)
            Spec = np.abs(librosa.core.stft(y=Xin, n_fft=frameSize, hop_length=frameShift, center=False))
            H_Spec, P_Spec = librosa.decompose.hpss(S=Spec)
            fv_H = librosa.feature.melspectrogram(S=H_Spec, n_mels=PARAMS['n_mels'])
            fv_P = librosa.feature.melspectrogram(S=P_Spec, n_mels=PARAMS['n_mels'])
            fv_H = fv_H.astype(np.float32)
            fv_P = fv_P.astype(np.float32)
            fv = np.append(fv_H, fv_P, axis=0) 

        if PARAMS['save_flag']:
            if not os.path.exists(feature_opDir+'/'+classname+'/'):
                os.makedirs(feature_opDir+'/'+classname+'/')
            np.save(feature_opDir+'/'+classname+'/'+fName+'.npy', fv)
    else:
        fv = np.load(feature_opDir+'/'+classname+'/'+fName+'.npy', allow_pickle=True)
    
    return fv

