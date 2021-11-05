#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:52:56 2021

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati

"""

import numpy as np
cimport numpy as np
from scipy.signal import medfilt
from scipy.stats import skew, kurtosis


np.import_array()  # needed to initialize numpy-API




def extract_patches(FV, shape, patch_size, patch_shift):
    cdef int frmStart, frmEnd, i
    cdef int nFrames = shape[1]
    cdef int half_win = int(patch_size/2)
    cdef int numPatches = len(list(range(half_win, nFrames-half_win, patch_shift)))
    cdef np.ndarray[double, ndim=3] patches = np.zeros((numPatches, shape[0], patch_size))
    cdef int nPatch = 0
    cdef int numFrame = 0
    for i in range(half_win, nFrames-half_win, patch_shift):
        frmStart = i-half_win
        # frmEnd = i+half_win
        frmEnd = np.min([frmStart + patch_size, nFrames])
        if (frmEnd-frmStart)<patch_size:
            frmStart = frmEnd-patch_size
        patches[nPatch,:,:] = FV[:,frmStart:frmEnd].copy()
        nPatch += 1
        numFrame += patch_shift
    return patches



def removeSilence(Xin, nSamples, energy, nFrames, fs, Tw, Ts, alpha=0.025, beta=0.075):
    '''
    Remove silence regions from audio files

    Parameters
    ----------
    Xin : array
        Audio signal.
    nSamples : int
        Number of samples in the signal.
    energy : array
        Root squared mean energy per frame.
    nFrames : int
        Number of frames.
    fs : int
        Sampling rate.
    Tw : int
        Short-term frame size (in miliseconds).
    Ts : int
        Short-term frame shift (in miliseconds).
    alpha : float, optional
        Parameter that controls the energy threshold to determine silent 
        frames. The default is 0.025.
    beta : float, optional
        Parameter that determines the minimum duration of a silent 
        segment (in seconds). The default is 0.075.

    Returns
    -------
    Xin_silrem : array
        Silence removed audio signal.
    sample_silMarker : array
        Silence/Non-silence annotation for every sample in the original audio
        signal.
    frame_silMarker : array
        Silence/Non-silence annotation for every short-term frame in the 
        original audio signal.
    totalSilDuration : int
        Duration of all silence segments.

    '''
    cdef int frameSize = int((Tw*fs)/1000) # Frame size in number of samples
    cdef int frameShift = int((Ts*fs)/1000) # Frame shift in number of samples
    cdef int totalSilDuration = 0
    cdef float energyThresh = alpha*np.max(energy) # TEST WITH MEAN
    cdef int i, j, k, l, nSil
    cdef np.ndarray[double, ndim=1] frame_silMarker_dbl = np.zeros((nFrames,))
    cdef np.ndarray[long, ndim=1] frame_silMarker = np.zeros((nFrames,)).astype(int)
    cdef np.ndarray[long, ndim=2] silences = np.zeros((nFrames,2)).astype(int)
    cdef np.ndarray[long, ndim=1] sample_silMarker = np.ones((nSamples,)).astype(int)
    cdef np.ndarray[float, ndim=1] Xin_silrem = np.ones((nSamples,)).astype(np.float32)

    frame_silMarker[energy < energyThresh] = 0
    frame_silMarker[energy >= energyThresh] = 1
    
    # Removing spurious detections
    frame_silMarker_dbl = medfilt(frame_silMarker.flatten(), 5)
    frame_silMarker = (frame_silMarker_dbl>0.5).astype(int)

    i=0
    nSil = 0
    while i<nFrames:
        while frame_silMarker[i]==1:
            if i == nFrames-1:
                break
            i = i + 1
        j = i
        while frame_silMarker[j]==0:
            if j == nFrames-1:
                break
            j = j + 1
        k = np.max([frameShift*(i-1)+frameSize, 1])
        l = np.min([frameShift*(j-1)+frameSize,nSamples]);
        
        # Only silence segments of durations greater than given beta
        # (e.g. 100ms) are removed
        if (l-k)/fs > beta:
            sample_silMarker[k:l] = 0
            silences[nSil, :] = [k,l]
            nSil += 1
            totalSilDuration += (l-k)/fs
        i = j + 1
    
    numNonSil = 0
    if nSil>1:
        nonSilIdx = np.squeeze(np.where(sample_silMarker==1))
        numNonSil = len(nonSilIdx)
        Xin_silrem[:numNonSil] = Xin[nonSilIdx]
    else:
        numNonSil = nSamples
        Xin_silrem = Xin
        
    return Xin_silrem, sample_silMarker, frame_silMarker, totalSilDuration



def scale_data(FV, mean, stdev):
    '''
    Mean-variance scaling of the data.

    Parameters
    ----------
    FV : array
        Feature Vector (num_features x num_frames).
    mean : array
        Frame mean.
    stdev : array
        Frame standard-deviation.

    Returns
    -------
    FV_scaled : array
        Scaled feature vector.

    '''
    cdef np.ndarray[double, ndim=2] M = np.zeros(np.shape(FV))
    cdef np.ndarray[double, ndim=2] S = np.zeros(np.shape(FV))
    cdef np.ndarray[double, ndim=2] FV_scaled = FV.copy().astype(np.float64)
    cdef int i=0
    M = np.repeat(np.array(mean, ndmin=2, dtype=np.float64).T, np.shape(M)[1], axis=1)
    S = np.repeat(np.array(stdev, ndmin=2, dtype=np.float64).T, np.shape(S)[1], axis=1)
    FV_scaled = np.subtract(FV_scaled, M)
    FV_scaled = np.divide(FV_scaled, S+1e-10)
    
    return FV_scaled


def get_data_statistics(FV, stat_type='skew', axis=0):
    '''
    Generate data statistics along the given axis

    Parameters
    ----------
    FV : array (Nxfxt), N patches of size fxt
        Feature Vector.
    stat_type : string
        Statistic type. The default is 'skew'. Other options are 'mean',
        'variance', and 'kurtosis'.
    axis : int, optional
        Axis along which skewness is to be computed. axis=0 indicates along 
        the columns (percussive), while axis=1 indicates along the rows 
        (harmonic). The default is 0.

    Returns
    -------
    Stat : array (Nxf) of (Nxt)
        Statistics vector along given axis.

    '''
    cdef int i = 0, nPatches = np.shape(FV)[0]
    cdef np.ndarray[double, ndim=2] Stat_harm = np.zeros((nPatches, np.shape(FV)[1])) # axis=1
    cdef np.ndarray[double, ndim=2] Stat_perc = np.zeros((nPatches, np.shape(FV)[2])) # axis=0
    for i in range(nPatches):
        if stat_type=='mean':
            stat_vec = np.mean(np.squeeze(FV[i,:,:]), axis=axis)
        if stat_type=='variance':
            stat_vec = np.var(np.squeeze(FV[i,:,:]), axis=axis)
        if stat_type=='skew':
            stat_vec = skew(np.squeeze(FV[i,:,:]), axis=axis)
        if stat_type=='kurtosis':
            stat_vec = kurtosis(np.squeeze(FV[i,:,:]), axis=axis)
        if axis==0: # percussive
            Stat_perc[i,:] = stat_vec
        else: # harmonic
            Stat_harm[i,:] = stat_vec

    if axis==0:
        return Stat_perc
    else:
        return Stat_harm
    
