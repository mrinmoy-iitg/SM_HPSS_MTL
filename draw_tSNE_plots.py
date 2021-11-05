#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:47:58 2020

@author: Mrinmoy Bhattacharjee, PhD Scholar, EEE Dept., IIT Guwahati
"""

import os
import datetime
import lib.misc as misc
import numpy as np
import lib.preprocessing as preproc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler
import sys
from lib.cython_impl.tools import get_data_statistics as cget_data_statistics
from sklearn.cluster import KMeans, MiniBatchKMeans




def load_data(PARAMS, folder, file_list, target_dB=None):
    np.random.shuffle(file_list['speech'])
    np.random.shuffle(file_list['music'])

    file_list_sp = file_list['speech'].copy()
    file_list_mu = file_list['music'].copy()

    Data_sp = np.empty([], dtype=float)
    Data_mu = np.empty([], dtype=float)
    Data_spmu = np.empty([], dtype=float)

    if not os.path.exists(PARAMS['feature_opDir']+'/speech/'):
        os.makedirs(PARAMS['feature_opDir']+'/speech/')
    if not os.path.exists(PARAMS['feature_opDir']+'/music/'):
        os.makedirs(PARAMS['feature_opDir']+'/music/')

    if len(PARAMS['classes'])==3:
        np.random.shuffle(file_list['speech+music'])
        file_list_spmu = file_list['speech+music'].copy()
        if not os.path.exists(PARAMS['feature_opDir']+'/speech_music/'):
            os.makedirs(PARAMS['feature_opDir']+'/speech_music/')

    n_fft = PARAMS['n_fft'][PARAMS['Model']]
    n_mels = PARAMS['n_mels'][PARAMS['Model']]
    featName = PARAMS['featName'][PARAMS['Model']]

    if not os.path.exists(PARAMS['opDir_data'] + '/Data_sp' + '_' + featName + '.pkl'):
        fl_count = 0
        for sp_fName in file_list_sp:
            fl_count += 1
            sp_fName_path = folder + '/speech/' + sp_fName
            if not os.path.exists(sp_fName_path):
                continue
            fv_sp = preproc.get_featuregram(PARAMS, 'speech', PARAMS['feature_opDir'], sp_fName_path, '', None, n_fft, n_mels, featName)
            fv_sp_patches = preproc.get_feature_patches(PARAMS, fv_sp, PARAMS['W'], PARAMS['W_shift'], featName)
            n_dims = np.shape(fv_sp_patches)[1]
            if PARAMS['striation_type']:

                if PARAMS['striation_type']=='Row':
                    if 'HarmPerc' in featName:
                        fv_sp_patches_H = cget_data_statistics(fv_sp_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=1) # Row
                        # fv_sp_patches_P = cget_data_statistics(fv_sp_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=1) # Row
                        # fv_sp_patches = np.append(fv_sp_patches_H, fv_sp_patches_P, axis=1)
                        fv_sp_patches = fv_sp_patches_H
                    else:
                        fv_sp_patches = cget_data_statistics(fv_sp_patches, stat_type=PARAMS['stat_type'], axis=1) # Row
                        
                elif PARAMS['striation_type']=='Col':
                    if 'HarmPerc' in featName:
                        # fv_sp_patches_H = cget_data_statistics(fv_sp_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=0) # Col
                        fv_sp_patches_P = cget_data_statistics(fv_sp_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=0) # Col
                        # fv_sp_patches = np.append(fv_sp_patches_H, fv_sp_patches_P, axis=1)
                        fv_sp_patches = fv_sp_patches_P
                    else:
                        fv_sp_patches = cget_data_statistics(fv_sp_patches, stat_type=PARAMS['stat_type'], axis=0) # Col
                
                elif PARAMS['striation_type']=='RowCol':
                    if 'HarmPerc' in featName:
                        fv_sp_patches_H_r = cget_data_statistics(fv_sp_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=1) # Row
                        # fv_sp_patches_H_c = cget_data_statistics(fv_sp_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=0) # Col
                        # fv_sp_patches_P_r = cget_data_statistics(fv_sp_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=1) # Row
                        fv_sp_patches_P_c = cget_data_statistics(fv_sp_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=0) # Col
                        # fv_sp_patches = np.append(fv_sp_patches_H_r, fv_sp_patches_H_c, axis=1)
                        # fv_sp_patches = np.append(fv_sp_patches, fv_sp_patches_P_r, axis=1)
                        # fv_sp_patches = np.append(fv_sp_patches, fv_sp_patches_P_c, axis=1)
                        fv_sp_patches = np.append(fv_sp_patches_H_r, fv_sp_patches_P_c, axis=1)
                    else:
                        fv_sp_patches_H = cget_data_statistics(fv_sp_patches, stat_type=PARAMS['stat_type'], axis=1) # Row
                        fv_sp_patches_P = cget_data_statistics(fv_sp_patches, stat_type=PARAMS['stat_type'], axis=0) # Col
                        fv_sp_patches = np.append(fv_sp_patches_H, fv_sp_patches_P, axis=1)
                    
            if np.size(Data_sp)<=1:
                Data_sp = fv_sp_patches
            else:
                Data_sp = np.append(Data_sp, fv_sp_patches, axis=0)
            print(fl_count, '/', len(file_list_sp), 'Speech data: ', np.shape(Data_sp), np.shape(fv_sp_patches))
        if PARAMS['save_flag']:
            misc.save_obj(Data_sp, PARAMS['opDir_data'], 'Data_sp' + '_' + featName)
    else:
        Data_sp = misc.load_obj(PARAMS['opDir_data'], 'Data_sp' + '_' + featName)
    print('Speech data: ', np.shape(Data_sp))


    if not os.path.exists(PARAMS['opDir_data'] + '/Data_mu' + '_' + featName + '.pkl'):
        fl_count = 0
        for mu_fName in file_list_mu:
            fl_count += 1
            mu_fName_path = folder + '/music/' + mu_fName
            if not os.path.exists(mu_fName_path):
                continue
            fv_mu = preproc.get_featuregram(PARAMS, 'music', PARAMS['feature_opDir'], '', mu_fName_path, None, n_fft, n_mels, featName)
            fv_mu_patches = preproc.get_feature_patches(PARAMS, fv_mu, PARAMS['W'], PARAMS['W_shift'], featName)
            n_dims = np.shape(fv_mu_patches)[1]
            if PARAMS['striation_type']:
                if PARAMS['striation_type']=='Row':
                    if 'HarmPerc' in featName:
                        fv_mu_patches_H = cget_data_statistics(fv_mu_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=1) # Row
                        # fv_mu_patches_P = cget_data_statistics(fv_mu_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=1) # Row
                        # fv_mu_patches = np.append(fv_mu_patches_H, fv_mu_patches_P, axis=1)
                        fv_mu_patches = fv_mu_patches_H
                    else:
                        fv_mu_patches = cget_data_statistics(fv_mu_patches, stat_type=PARAMS['stat_type'], axis=1) # Row
                        
                elif PARAMS['striation_type']=='Col':
                    if 'HarmPerc' in featName:
                        # fv_mu_patches_H = cget_data_statistics(fv_mu_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=0) # Col
                        fv_mu_patches_P = cget_data_statistics(fv_mu_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=0) # Col
                        # fv_mu_patches = np.append(fv_mu_patches_H, fv_mu_patches_P, axis=1)
                        fv_mu_patches = fv_mu_patches_P
                    else:
                        fv_mu_patches = cget_data_statistics(fv_mu_patches, stat_type=PARAMS['stat_type'], axis=0) # Col
                elif PARAMS['striation_type']=='RowCol':
                    if 'HarmPerc' in featName:
                        fv_mu_patches_H_r = cget_data_statistics(fv_mu_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=1) # Row
                        # fv_mu_patches_H_c = cget_data_statistics(fv_mu_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=0) # Col
                        # fv_mu_patches_P_r = cget_data_statistics(fv_mu_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=1) # Row
                        fv_mu_patches_P_c = cget_data_statistics(fv_mu_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=0) # Col
                        # fv_mu_patches = np.append(fv_mu_patches_H_r, fv_mu_patches_H_c, axis=1)
                        # fv_mu_patches = np.append(fv_mu_patches, fv_mu_patches_P_r, axis=1)
                        # fv_mu_patches = np.append(fv_mu_patches, fv_mu_patches_P_c, axis=1)
                        fv_mu_patches = np.append(fv_mu_patches_H_r, fv_mu_patches_P_c, axis=1)
                    else:
                        fv_mu_patches_H = cget_data_statistics(fv_mu_patches, stat_type=PARAMS['stat_type'], axis=1) # Row
                        fv_mu_patches_P = cget_data_statistics(fv_mu_patches, stat_type=PARAMS['stat_type'], axis=0) # Col
                        fv_mu_patches = np.append(fv_mu_patches_H, fv_mu_patches_P, axis=1)
                    
            if np.size(Data_mu)<=1:
                Data_mu = fv_mu_patches
            else:
                Data_mu = np.append(Data_mu, fv_mu_patches, axis=0)
            print(fl_count, '/', len(file_list_mu), 'Music data: ', np.shape(Data_mu), np.shape(fv_mu_patches))
        if PARAMS['save_flag']:
            misc.save_obj(Data_mu, PARAMS['opDir_data'], 'Data_mu' + '_' + featName)
    else:
        Data_mu = misc.load_obj(PARAMS['opDir_data'], 'Data_mu' + '_' + featName)
    print('Music data: ', np.shape(Data_mu))

    if len(PARAMS['classes'])==3:
        if not os.path.exists(PARAMS['opDir_data'] + '/Data_spmu' + '_' + featName + '.pkl'):
            fl_count = 0
            for spmu_fName in file_list_spmu:
                fl_count += 1
                spmu_info = file_list_spmu.pop()
                sp_fName = spmu_info['speech']
                sp_fName_path = folder + '/speech/' + sp_fName
                mu_fName = spmu_info['music']
                mu_fName_path = folder + '/music/' + mu_fName
                if not target_dB:
                    target_dB = spmu_info['SMR']
                if (not os.path.exists(mu_fName_path)) or (not os.path.exists(sp_fName_path)):
                    continue
                fv_spmu = preproc.get_featuregram(PARAMS, 'speech_music', PARAMS['feature_opDir'], sp_fName_path, mu_fName_path, target_dB, n_fft, n_mels, featName)
                fv_spmu_patches = preproc.get_feature_patches(PARAMS, fv_spmu, PARAMS['W'], PARAMS['W_shift'], featName)
                n_dims = np.shape(fv_spmu_patches)[1]
                if PARAMS['striation_type']:
                    if PARAMS['striation_type']=='Row':
                        if 'HarmPerc' in featName:
                            fv_spmu_patches_H = cget_data_statistics(fv_spmu_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=1) # Row
                            # fv_spmu_patches_P = cget_data_statistics(fv_spmu_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=1) # Row
                            # fv_spmu_patches = np.append(fv_spmu_patches_H, fv_spmu_patches_P, axis=1)
                            fv_mu_patches = fv_mu_patches_H
                        else:
                            fv_spmu_patches = cget_data_statistics(fv_spmu_patches, stat_type=PARAMS['stat_type'], axis=1) # Row
                            
                    elif PARAMS['striation_type']=='Col':
                        if 'HarmPerc' in featName:
                            # fv_spmu_patches_H = cget_data_statistics(fv_spmu_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=0) # Col
                            fv_spmu_patches_P = cget_data_statistics(fv_spmu_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=0) # Col
                            # fv_spmu_patches = np.append(fv_spmu_patches_H, fv_spmu_patches_P, axis=1)
                            fv_mu_patches = fv_mu_patches_P
                        else:
                            fv_spmu_patches = cget_data_statistics(fv_spmu_patches, stat_type=PARAMS['stat_type'], axis=0) # Col
                    elif PARAMS['striation_type']=='RowCol':
                        if 'HarmPerc' in featName:
                            fv_spmu_patches_H_r = cget_data_statistics(fv_spmu_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=1) # Row
                            # fv_spmu_patches_H_c = cget_data_statistics(fv_spmu_patches[:,:int(n_dims/2),:], stat_type=PARAMS['stat_type'], axis=0) # Col
                            # fv_spmu_patches_P_r = cget_data_statistics(fv_spmu_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=1) # Row
                            fv_spmu_patches_P_c = cget_data_statistics(fv_spmu_patches[:,int(n_dims/2):,:], stat_type=PARAMS['stat_type'], axis=0) # Col
                            # fv_spmu_patches = np.append(fv_spmu_patches_H_r, fv_spmu_patches_H_c, axis=1)
                            # fv_spmu_patches = np.append(fv_spmu_patches, fv_spmu_patches_P_r, axis=1)
                            # fv_spmu_patches = np.append(fv_spmu_patches, fv_spmu_patches_P_c, axis=1)
                            fv_spmu_patches = np.append(fv_spmu_patches_H_r, fv_spmu_patches_P_c, axis=1)
                        else:
                            fv_spmu_patches_H = cget_data_statistics(fv_spmu_patches, stat_type=PARAMS['stat_type'], axis=1) # Row
                            fv_spmu_patches_P = cget_data_statistics(fv_spmu_patches, stat_type=PARAMS['stat_type'], axis=0) # Col
                            fv_spmu_patches = np.append(fv_spmu_patches_H, fv_spmu_patches_P, axis=1)

                if np.size(Data_spmu)<=1:
                    Data_spmu = fv_spmu_patches
                else:
                    Data_spmu = np.append(Data_spmu, fv_spmu_patches, axis=0)
                print(fl_count, '/', len(file_list_spmu), 'Speech-Music data: ', np.shape(Data_spmu), np.shape(fv_spmu_patches))
            if PARAMS['save_flag']:
                misc.save_obj(Data_spmu, PARAMS['opDir_data'], 'Data_spmu' + '_' + featName)
        else:
            Data_spmu = misc.load_obj(PARAMS['opDir_data'], 'Data_spmu' + '_' + featName)
        print('Speech-Music data: ', np.shape(Data_spmu))

    return Data_mu, Data_sp, Data_spmu



def get_train_test_files(cv_file_list, numCV, foldNum):
    train_files = {}
    test_files = {}
    classes = {'music':0, 'speech':1, 'speech+music':2}
    for class_name in classes.keys():
        train_files[class_name] = []
        test_files[class_name] = []
        for i in range(numCV):
            files = cv_file_list[class_name]['fold'+str(i)]
            if foldNum==i:
                test_files[class_name].extend(files)
            else:
                train_files[class_name].extend(files)
    
    return train_files, test_files




def plot_figure(fName, Data_embedded, Label):
    colors = ['b', 'm', 'k']
    markers = ['o', 'v', '+']
    Cl = np.unique(Label)
    plt.figure()
    for i in range(len(Cl)):
        lab = int(Cl[i])
        idx = np.squeeze(np.where(Label==lab))
        print(len(idx), lab)
        plt.scatter(
            x=Data_embedded[idx,0],
            y=Data_embedded[idx,1],
            c=colors[i],
            edgecolors=colors[i],
            alpha=1,
            marker=markers[i],
            s=20,
            lw = 1,
            label=PARAMS['classes'][Cl[i]],
            )
        plt.draw()
    plt.legend(prop={'size': 10})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = '14'
    # plt.show(block=True)
    plt.savefig(fName, bbox_inches='tight')
    


def grid_search_tSNE_params(PARAMS, n_components, Data):
    for P in range(5,51,5):
        for E in range(2,15,2):
            for L in range(50,251,50):
                tsne = TSNE(
                    n_components=n_components,
                    perplexity=P,
                    early_exaggeration=E,
                    learning_rate=L,
                    metric='euclidean',
                    verbose=1,
                    n_jobs=3,
                    square_distances=True,
                    )                    
                Data_embedded = tsne.fit_transform(Data)
                print('Data_embedded: ', np.shape(Data_embedded), np.shape(Data))

                if PARAMS['plot_fig']:
                    opDir_fig = PARAMS['opDir'] + '/figures/'
                    if not os.path.exists(opDir_fig):
                        os.makedirs(opDir_fig)
                    fName = opDir_fig +'/tSNE_plot_' + PARAMS['stat_type'] + '_P' + str(P) + '_E' + str(E) + '_L' + str(L) + '.jpg'
                    plot_figure(fName, Data_embedded, Label)




def remove_outliers(X, k=3):
    mn = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    inlier_idx = []
    for i in range(np.shape(X)[0]):
        x0 = np.subtract(X[i,:], mn)
        if np.sqrt(np.sum(x0**2))<=k*np.sum(sd):
            inlier_idx.append(i)
    
    return inlier_idx
    
    
    

def __init__():
    patch_size = 68
    patch_shift = 6
    PARAMS = {
            'today': datetime.datetime.now().strftime("%Y-%m-%d"),
            'folder': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/GTZAN', # Laptop
            # 'folder':'/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Scheirer-slaney/', # Laptop
            # 'folder':'/media/mrinmoy/NTFS_Volume/Phd_Work/Data/Movie_Music_Speech_Noise_Mix_Corpus_Movie-MUSNOMIX/MUSNOMIX_WAV/', # Laptop
            # 'folder': '/media/mrinmoy/NTFS_Volume/Phd_Work/Data/musan', # Laptop
            'feature_folder': '/home/mrinmoy/Documents/PhD_Work/Features/SpMu_HPSS_MTL/', # Laptop

            # 'folder': '/scratch/mbhattacharjee/data/musan/', # PARAMS ISHAN
            # 'feature_folder': './features/', # PARAM-ISHAN

            # 'folder': '/home/phd/mrinmoy.bhattacharjee/data/musan/', # EEE GPU
            # 'feature_folder': '/home1/PhD/mrinmoy.bhattacharjee/features/',  # EEE GPU

            'CV_folds': 3,
            'fold': 0,
            'save_flag': True,
            'Model': 'Lemaire_et_al',
            'classes': {0:'music', 1:'speech'}, # {0:'music', 1:'speech', 2:'speech_music'}
            'W':patch_size,
            'W_shift':patch_shift,
            'Tw': 25,
            'Ts': 10,
            'n_fft': {'Lemaire_et_al':400},
            'n_mels': {'Lemaire_et_al':21},
            'l_harm': {'Lemaire_et_al':21},
            'l_perc': {'Lemaire_et_al':11},
            'featName': {
                # 'Lemaire_et_al': 'MelSpec' , # 'LogMelSpec',
                'Lemaire_et_al': 'MelHarmPercSpec', #'LogMelHarmPercSpec',
                },
            'all_striations': ['Row', 'Col', 'RowCol'], # ['Row', 'Col', 'RowCol'], # 'Row', 'Col', 'RowCol', None
            'all_stats': ['skew'], # ['mean', 'variance', 'skew', 'kurtosis'],
            'plot_fig': True,
            'frame_level_scaling': False,
            'grid_search': False,
            # 'Perplexity':5,
            # 'Early_Exaggeration': 12,
            # 'Learning_Rate': 50,
            'n_jobs': 3,
            'n_clusters': 1000,
            }

    PARAMS['dataset_name_train'] = list(filter(None,PARAMS['folder'].split('/')))[-1]
    PARAMS['CV_path_train'] = './cross_validation_info/' + PARAMS['dataset_name_train'] + '/'    
    if not os.path.exists(PARAMS['CV_path_train']+'/cv_file_list.pkl'):
        print('Fold division of files for cross-validation not done!')
        sys.exit(0)
    PARAMS['cv_file_list'] = misc.load_obj(PARAMS['CV_path_train'], 'cv_file_list')

    return PARAMS




if __name__ == '__main__':
    PARAMS = __init__()

    PARAMS['feature_opDir'] = PARAMS['feature_folder'] + PARAMS['dataset_name_train'] + '/' + PARAMS['Model'] + '/' + PARAMS['featName'][PARAMS['Model']] + '/'
    if not os.path.exists(PARAMS['feature_opDir']):
        os.makedirs(PARAMS['feature_opDir'])
        
    for PARAMS['striation_type'] in PARAMS['all_striations']:
        PARAMS['opDir'] = './results/' + PARAMS['dataset_name_train'] + '/t-SNE/' + PARAMS['Model'] + '/' + PARAMS['today'] + '/' + PARAMS['featName'][PARAMS['Model']] + '/' + PARAMS['striation_type'] + '_' + str(len(PARAMS['classes'])) + 'classes/'
        if not os.path.exists(PARAMS['opDir']):
            os.makedirs(PARAMS['opDir'])
                    
        misc.print_configuration(PARAMS)
        print('featName: ', PARAMS['featName'][PARAMS['Model']], PARAMS['striation_type'])
        

        for PARAMS['stat_type'] in PARAMS['all_stats']:
            PARAMS['opDir_data'] = PARAMS['opDir']+'/__data_' + PARAMS['stat_type'] + '/'
            if not os.path.exists(PARAMS['opDir_data']):
                os.makedirs(PARAMS['opDir_data'])
        
            PARAMS['opDir_embeddings'] = PARAMS['opDir']+'/__embeddings_' + PARAMS['stat_type'] + '/'
            if not os.path.exists(PARAMS['opDir_embeddings']):
                os.makedirs(PARAMS['opDir_embeddings'])
                        
            for PARAMS['fold'] in range(0,1): # range(PARAMS['CV_folds']):
                PARAMS['train_files'], PARAMS['test_files'] = get_train_test_files(PARAMS['cv_file_list'], PARAMS['CV_folds'],  PARAMS['fold'])
                matFileName = 'Data_embedding_' + PARAMS['featName'][PARAMS['Model']] + '_fold' + str(PARAMS['fold']) +'.mat'
                
                Data_mu, Data_sp, Data_spmu = load_data(PARAMS, PARAMS['folder'], PARAMS['train_files'])
                
                fig_fName = PARAMS['opDir'] + '/../' + PARAMS['striation_type'] + '_' + str(len(PARAMS['classes'])) + 'classes_tSNE_plot_' + PARAMS['stat_type'] + '_fold' + str(PARAMS['fold']) + '.jpg'
                if not os.path.exists(PARAMS['opDir_embeddings'] + '/' + matFileName.split('.')[0] + '.pkl'):

                    Data = np.append(Data_mu, Data_sp, axis=0)
                    if len(PARAMS['classes'])==3:
                        Data = np.append(Data, Data_spmu, axis=0)
                    
                    print('Data: ', np.shape(Data))
                    Data = StandardScaler().fit_transform(Data)
                    print('Data (after scaling): ', np.shape(Data))

                    Data_mu = Data[:np.shape(Data_mu)[0], :]
                    Data_sp = Data[np.shape(Data_mu)[0]:, :]
                    if len(PARAMS['classes'])==3:
                        Data_spmu = Data[np.shape(Data_mu)[0]+np.shape(Data_sp)[0]:, :]
                    
                    kmeans_mu = KMeans(n_clusters=PARAMS['n_clusters'], verbose=1, algorithm='full').fit(Data_mu)
                    Data_mu = kmeans_mu.cluster_centers_
    
                    kmeans_sp = KMeans(n_clusters=PARAMS['n_clusters'], verbose=1, algorithm='full').fit(Data_sp)
                    Data_sp = kmeans_sp.cluster_centers_
    
                    if len(PARAMS['classes'])==3:
                        kmeans_spmu = KMeans(n_clusters=PARAMS['n_clusters'], verbose=1, algorithm='full').fit(Data_spmu)
                        Data_spmu = kmeans_spmu.cluster_centers_
                                        
                    # print('With outlier mu: ', np.shape(Data_mu))
                    # inlier_idx = remove_outliers(Data_mu)
                    # Data_mu = Data_mu[inlier_idx,:]
                    # print('Without outlier mu: ', np.shape(Data_mu))
    
                    # print('With outlier sp: ', np.shape(Data_sp))
                    # inlier_idx = remove_outliers(Data_sp)
                    # Data_sp = Data_sp[inlier_idx,:]
                    # print('Without outlier sp: ', np.shape(Data_sp))
    
                    # if len(PARAMS['classes'])==3:
                    #     print('With outlier spmu: ', np.shape(Data_spmu))
                    #     inlier_idx = remove_outliers(Data_spmu)
                    #     Data_spmu = Data_spmu[inlier_idx,:]
                    #     print('Without outlier spmu: ', np.shape(Data_spmu))

                    Data = np.append(Data_mu, Data_sp, axis=0)
                    if len(PARAMS['classes'])==3:
                        Data = np.append(Data, Data_spmu, axis=0)
                                        
                    Label = np.zeros(np.shape(Data_mu)[0])
                    Label = np.append(Label, np.ones(np.shape(Data_sp)[0]))
                    if len(PARAMS['classes'])==3:
                        Label = np.append(Label, np.ones(np.shape(Data_spmu)[0])*2)
                    Label = Label.astype(int)
                    print('Label: ', np.shape(Label))

                    
                    n_components = len(np.unique(Label))
                    print('n_components: ', n_components)
                    
                    if PARAMS['grid_search']:
                        grid_search_tSNE_params(PARAMS, n_components, Data)
        
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=30.0,
                        early_exaggeration=12.0,
                        learning_rate=200.0,
                        n_iter=1000,
                        n_iter_without_progress=300,
                        min_grad_norm=1e-07,
                        metric='sqeuclidean',
                        init='pca',
                        verbose=1,
                        random_state=None,
                        # method='exact',
                        method='barnes_hut',
                        angle=0.5,
                        n_jobs=10,
                        square_distances=True
                        )                    
    
                    Data_embedded = tsne.fit_transform(Data, y=Label)
                    print('Data_embedded: ', np.shape(Data_embedded), np.shape(Data))
                    savemat(PARAMS['opDir_embeddings']+matFileName, {'Data':Data_embedded, 'Label':Label})
                    misc.save_obj({'Data_embedded':Data_embedded, 'Label':Label}, PARAMS['opDir_embeddings'], matFileName.split('.')[0])
                else:
                    Data_embedded = misc.load_obj(PARAMS['opDir_embeddings'], matFileName.split('.')[0])['Data_embedded']
                    Label = misc.load_obj(PARAMS['opDir_embeddings'], matFileName.split('.')[0])['Label']

                print('With outlier: ', np.shape(Data_embedded))
                inlier_idx = remove_outliers(Data_embedded, k=1)
                Data_embedded = Data_embedded[inlier_idx,:]
                Label = Label[inlier_idx]
                print('Without outlier: ', np.shape(Data_embedded))
                        
                if PARAMS['plot_fig']:
                    plot_figure(fig_fName, Data_embedded, Label)
                    