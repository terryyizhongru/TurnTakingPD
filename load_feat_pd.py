
import os
import glob
from pathlib import Path
import pdb

import numpy as np
from tqdm import tqdm
import pandas as pd

from scipy.stats import shapiro, kruskal, kstest, norm, levene



demo_data_file = '/home/yzhong/gits/TurnTakingPD/demogr_perpp.txt'
ID2EMO = {}
with open(demo_data_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        datas = line.split('\t')
        ID2EMO[datas[0]] = datas[1:]
        

# np.set_printoptions(threshold=np.inf)


def get_demo(filename):
    subject_id = filename.split('_')[0][-4:]
    group_id = subject_id[:2]
    if group_id not in ['11', '21', '22']:
        raise ValueError(f"Invalid group id {group_id}")
    if subject_id in {'2219', '2123'}:
        return None, None, None
    if subject_id in {'2135'}:
        return subject_id, group_id, ['NA', 'NA', 'NA', 'NA']
    return subject_id, group_id, ID2EMO[subject_id]


def add2list(group_id, feature, ls):
    if group_id == '11':
        ls[0].append(feature)
    elif group_id == '21':
        ls[1].append(feature)
    elif group_id == '22':
        ls[2].append(feature)
    else:
        print(f'Invalid group id {group_id}')
        
       
   

def load_utt_feat(feature_name='energy', stats='mean'):
        # 3 sublists for YA OA PD
    exp2lists = {'BoundaryTone': [[], [], []], 'EarlyLate': [[], [], []], 'PictureNaming': [[], [], []]}
    avg_diff = []
    for folder in ['BoundaryTone', 'EarlyLate', 'PictureNaming']:
        feature_folder = os.path.join(base_folder_path, folder + '-features', feature_name)
        feature_folder = Path(feature_folder)
        npy_files = list(feature_folder.glob('*.npy'))
        print(f'Processing {folder} folder...')
        print(f'Found {len(npy_files)} npy files')

        cnt = 0
        for npy_file in npy_files:
            feature = np.load(npy_file)         
            # check if all 0 value
            if np.max(feature) == 0 and np.min(feature) == 0:
                cnt += 1
                continue
            
            group_id = get_group_id(npy_file.stem)
            feature = feature[feature != 0]
            if feature_name == 'f0':
                feature = feature[feature < 500.0]
                feature = np.log(feature)
                if feature.shape[0] == 0:
                    print(f'All value larger than 500.0 in {npy_file}')
                    continue
            if stats == 'mean':
                feature = np.mean(feature)
            elif stats == 'std':
                feature = np.std(feature)
            add2list(group_id, (npy_file.stem, feature), exp2lists[folder])
        print(f'{cnt} files with all 0 values')
        
    
    all3 = [[], [], []] 
    for i in range(3):
        all3[i] += exp2lists['BoundaryTone'][i] + exp2lists['EarlyLate'][i] + exp2lists['PictureNaming'][i]


        
    return all3, exp2lists
    

def load_rp():
        # 3 sublists for YA OA PD
    exp2lists = {'BoundaryTone': [[], [], []], 'EarlyLate': [[], [], []], 'PictureNaming': [[], [], []]}
    
    for folder in ['BoundaryTone', 'EarlyLate', 'PictureNaming']:
        feature_list = os.path.join('/home/yzhong/gits/TurnTakingPD/filelists', 'clean_id_responsetime_' + folder + '_filtered.txt')
        
        cnt = 0
        with open(feature_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                rp = float(line.strip().split('\t')[-1])
                basename = line.strip().split('\t')[0]
                group_id = get_group_id(basename)
                add2list(group_id, (basename, rp), exp2lists[folder])
                cnt += 1
                
        print(f'Processing {feature_list} ...')
        print(f'Found {cnt}')


    all3 = [[], [], []] 
    for i in range(3):
        all3[i] += exp2lists['BoundaryTone'][i] + exp2lists['EarlyLate'][i] + exp2lists['PictureNaming'][i]
    
    return all3, exp2lists




def load_feat(base_folder_path, feature_name='energy', log_value=False, YA=False, threeD=False):
        # 3 sublists for YA OA PD
    all_data = []
    all_data_3D = []

    exp2lists = {'BoundaryTone': [[], [], []], 'EarlyLate': [[], [], []], 'PictureNaming': [[], [], []]}
    avg_diff = []
    for exp_idx, folder in enumerate(['PictureNaming', 'EarlyLate', 'BoundaryTone']):
        feature_folder = os.path.join(base_folder_path, folder + '-features', feature_name)
        feature_folder = Path(feature_folder)
        npy_files = list(feature_folder.glob('*.npy'))
        print(f'Processing {folder} folder...')
        print(f'Found {len(npy_files)} npy files')

        cnt = 0
        for npy_file in npy_files:
            feature = np.load(npy_file)    
            # check if all 0 value
            # if nan in the feature
            if np.isnan(feature).any():
                print(f'nan in {npy_file}')
                # remove the nan value
                feature = feature[~np.isnan(feature)]
                if len(feature) == 0:
                    print(f'All nan value in {npy_file}')
                    continue
            if np.max(feature) == 0 and np.min(feature) == 0:
                cnt += 1
                continue
            
            subject_id, group_id, demo_data = get_demo(npy_file.stem)
            if subject_id is None:
                continue
            
        
            if feature_name == 'f0':
                feature = feature[feature != 0]

                feature = feature[feature < 500.0]
                if feature.shape[0] == 0:
                    print(f'All value larger than 500.0 in {npy_file}')
                    continue

            if log_value is True:              
                feature = np.log(feature)
            
            filename = npy_file.stem
            experiment = 'exp_' + str(exp_idx + 1) + '_' + folder
            if experiment.startswith('exp_1'):
                item = filename.split('_')[1].rstrip('.png')
            elif experiment.startswith('exp_3'):
                item = '_'.join(filename.split('_')[1:3])
            elif experiment.startswith('exp_2'):
                item = filename.split('_')[1]
            else:
                raise ValueError('Experiment not found')
            
            if demo_data[1] == 'NA':
                continue
                
            if YA is False and group_id == '11':
                continue
            
            if threeD is True:
                dim = feature.shape[0]
                if len(all_data_3D) == 0:
                    all_data_3D = [[] for _ in range(dim)]
                for i in range(dim):
                    utt = {
                    'experiment': 'exp_' + str(exp_idx + 1) + '_' + folder,
                    'group_id': group_id,
                    'value': feature[i],
                    'subject_id':subject_id,
                    'filename': filename.split('.')[0],
                    'item': item,
                    'age': demo_data[0],
                    'gender': demo_data[1],
                    'moca': demo_data[2],
                    'education': demo_data[3],
                    }
                    all_data_3D[i].append(utt)
            
            else:   
                    
                utt = {
                        'experiment': 'exp_' + str(exp_idx + 1) + '_' + folder,
                        'group_id': group_id,
                        'value': feature,
                        'subject_id':subject_id,
                        'filename': filename.split('.')[0],
                        'item': item,
                        'age': demo_data[0],
                        'gender': demo_data[1],
                        'moca': demo_data[2],
                        'education': demo_data[3],
                    }

                all_data.append(utt)
            
        print(f'{cnt} files with all 0 values')

    return all_data if threeD is False else all_data_3D



def basic_stats(group1, group2, groupnames=['OA(HC)', 'PD']):

    for i, group in enumerate([group1, group2]):
        # print(f'\n  stats of {groupnames[0]}')
        # # merge all data in subgroup
        # print(group.shape)
        # check if nan in subgroup
        if np.isnan(group).any():
            print('nan in subgroup')

        yield np.mean(group), np.std(group), np.median(group)
        


def stats_test(group1, group2, groupnames=['OA(HC)', 'PD']):


    d_stat, p_ks1 = kstest(group1, 'norm')
    # print(f"{groupnames[0]} kstest p-value: {p_ks1}")
 
    d_stat, p_ks2 = kstest(group2, 'norm')
    # print(f"{groupnames[1]} kstest p-value: {p_ks2}")
 
    stat, p_kw = kruskal(group1, group2)
    # print(f"Kruskal-Wallis test p value between group {groupnames[0]} and {groupnames[1]} : {p_kw}")

  
    levene_stat, p_lev = levene(group1, group2, center='median')
    # print(f"Levene test p value between group {groupnames[0]} and {groupnames[1]} : {p_lev}")

    return p_ks1, p_ks2, p_kw, p_lev



# all3_f0, exp2list_f0 = load_utt_feat(feature_name='f0')
# all3_f0_var, exp2list_f0_var = load_utt_feat(feature_name='f0', stats='std')

# all3_energy, exp2list_energy = load_utt_feat(feature_name='energy')
# all3_energy_var, exp2list_energy_var = load_utt_feat(feature_name='energy', stats='std')

# all3_rp, exp2list_rp = load_rp()
        


def all_level_analysis_frame(df):

  
    item = {}
    heads = ['level', 'OA_values', 'PD_values', 'OA_mean', 'PD_mean', 'OA_std', 'PD_std', 'OA_median', 'PD_median', 'p_ks1', 'p_ks2', 'p_kw', 'p_lev']
    result_datas = []
    
    # frame level analysis
    OA_values = np.concatenate(df[df['group_id'] == '21']['value'].values)
    PD_values = np.concatenate(df[df['group_id'] == '22']['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['frame', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())

    # utterance level
    # mean
    df_utt_mean = df.copy()
    df_utt_mean['value'] = df_utt_mean['value'].apply(lambda x: np.mean(x))
    OA_values = np.array(df_utt_mean[df_utt_mean['group_id'] == '21']['value'].values)
    PD_values = np.array(df_utt_mean[df_utt_mean['group_id'] == '22']['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())
    

    # std
    df_utt_std = df.copy()
    df_utt_std['value'] = df_utt_std['value'].apply(lambda x: np.std(x))
    OA_values = np.array(df_utt_std[df_utt_std['group_id'] == '21']['value'].values)
    PD_values = np.array(df_utt_std[df_utt_std['group_id'] == '22']['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance_std', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())
    
    # utterance level in each gender
    # mean, male
    df_utt_mean = df.copy()
    df_utt_mean['value'] = df_utt_mean['value'].apply(lambda x: np.mean(x))
    df_male_OA = df_utt_mean[(df_utt_mean['group_id'] == '21') & (df_utt_mean['gender'] == 'M')]
    df_male_PD = df_utt_mean[(df_utt_mean['group_id'] == '22') & (df_utt_mean['gender'] == 'M')]
    OA_values = np.array(df_male_OA['value'].values)
    PD_values = np.array(df_male_PD['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance_mean_male', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # mean, female
    df_female_OA = df_utt_mean[(df_utt_mean['group_id'] == '21') & (df_utt_mean['gender'] == 'V')]
    df_female_PD = df_utt_mean[(df_utt_mean['group_id'] == '22') & (df_utt_mean['gender'] == 'V')]
    OA_values = np.array(df_female_OA['value'].values)
    PD_values = np.array(df_female_PD['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance_mean_female', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())
    
    # std, male
    df_utt_std = df.copy()
    df_utt_std['value'] = df_utt_std['value'].apply(lambda x: np.std(x))
    df_male_OA = df_utt_std[(df_utt_std['group_id'] == '21') & (df_utt_std['gender'] == 'M')]
    df_male_PD = df_utt_std[(df_utt_std['group_id'] == '22') & (df_utt_std['gender'] == 'M')]
    OA_values = np.array(df_male_OA['value'].values)
    PD_values = np.array(df_male_PD['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance_std_male', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # std, female
    df_female_OA = df_utt_std[(df_utt_std['group_id'] == '21') & (df_utt_std['gender'] == 'V')]
    df_female_PD = df_utt_std[(df_utt_std['group_id'] == '22') & (df_utt_std['gender'] == 'V')]
    OA_values = np.array(df_female_OA['value'].values)
    PD_values = np.array(df_female_PD['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance_std_female', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # person level
    # mean of mean of all utterances
    df_person_mean_mean = df.copy()
    df_person_mean_mean['value'] = df_person_mean_mean['value'].apply(lambda x: np.mean(x))
    OA_values = np.array(df_person_mean_mean[df_person_mean_mean['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.array(df_person_mean_mean[df_person_mean_mean['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['person_mean_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())
    
    # # std of mean of all utterances
    # df_person_std_mean= df.copy()
    # df_person_std_mean['value'] = df_person_std_mean['value'].apply(lambda x: np.mean(x))
    # OA_values = np.array(df_person_std_mean[df_person_std_mean['group_id'] == '21'].groupby('subject_id')['value'].std().values)
    # PD_values = np.array(df_person_std_mean[df_person_std_mean['group_id'] == '22'].groupby('subject_id')['value'].std().values)
    # OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    # p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    # reses = ['person_std_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # # assign the values to the item
    # for k, v in zip(heads, reses):
    #     item[k] = v
        
    # result_datas.append(item.copy())

    # mean of std of all utterances
    df_person_mean_std = df.copy()
    df_person_mean_std['value'] = df_person_mean_std['value'].apply(lambda x: np.std(x))
    OA_values = np.array(df_person_mean_std[df_person_mean_std['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.array(df_person_mean_std[df_person_mean_std['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['person_mean_std', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())
    
    # # std of std of all utterances
    # df_person_std_std = df.copy()
    # df_person_std_std['value'] = df_person_std_std['value'].apply(lambda x: np.std(x))
    # OA_values = np.array(df_person_std_std[df_person_std_std['group_id'] == '21'].groupby('subject_id')['value'].std().values)
    # PD_values = np.array(df_person_std_std[df_person_std_std['group_id'] == '22'].groupby('subject_id')['value'].std().values)
    # OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    # p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    # reses = ['person_std_std', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # # assign the values to the item
    # for k, v in zip(heads, reses):
    #     item[k] = v
        
    # result_datas.append(item.copy())

    # done person level in two gender
    # mean of mean of all male utterances
    df_male_mean_std = df.copy()
    df_male_mean_std = df_male_mean_std[df_male_mean_std['gender'] == 'M']
    df_male_mean_std['value'] = df_male_mean_std['value'].apply(lambda x: np.mean(x))
    OA_values = np.array(df_male_mean_std[df_male_mean_std['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.array(df_male_mean_std[df_male_mean_std['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['male_mean_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())

    # mean of mean of all female utterances
    df_female_mean_std = df.copy()
    df_female_mean_std = df_female_mean_std[df_female_mean_std['gender'] == 'V']
    df_female_mean_std['value'] = df_female_mean_std['value'].apply(lambda x: np.mean(x))
    OA_values = np.array(df_female_mean_std[df_female_mean_std['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.array(df_female_mean_std[df_female_mean_std['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['female_mean_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())

    # mean of std of all male utterances
    df_male_mean_std = df.copy()
    df_male_mean_std = df_male_mean_std[df_male_mean_std['gender'] == 'M']
    df_male_mean_std['value'] = df_male_mean_std['value'].apply(lambda x: np.std(x))
    OA_values = np.array(df_male_mean_std[df_male_mean_std['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.array(df_male_mean_std[df_male_mean_std['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['male_mean_std', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())

    # mean of std of all female utterances
    df_female_mean_std = df.copy()
    df_female_mean_std = df_female_mean_std[df_female_mean_std['gender'] == 'V']
    df_female_mean_std['value'] = df_female_mean_std['value'].apply(lambda x: np.std(x))
    OA_values = np.array(df_female_mean_std[df_female_mean_std['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.array(df_female_mean_std[df_female_mean_std['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['female_mean_std', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())    

    # # mean of mean of all person
    #   def mean_f0(series):
    #     return np.mean([np.mean(arr) for arr in series])


    # df_gender_mean_mean_mean = df.copy()
    # # remove the gender with 'NA' value
    # df_gender_mean_mean_mean = df_gender_mean_mean_mean[df_gender_mean_mean_mean['gender'] != 'NA']
    # # generate a new df that average the mean of all utterances for each person while maintaining the gender information
    # df_gender_mean_mean_mean = df_gender_mean_mean_mean.groupby(['subject_id', 'gender']).agg({
    #     'value': mean_f0,
    #     'group_id': 'first'
    #     }).reset_index()
    # # get average of the mean of all person for each gender
    # OA_values = np.array(df_gender_mean_mean_mean[df_gender_mean_mean_mean['group_id'] == '21'].groupby('gender')['value'].mean().values)
    # PD_values = np.array(df_gender_mean_mean_mean[df_gender_mean_mean_mean['group_id'] == '22'].groupby('gender')['value'].mean().values)
    # OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    # p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    # reses = ['gender_mean_mean_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # # assign the values to the item
    # for k, v in zip(heads, reses):
    #     item[k] = v
    
    # result_datas.append(item.copy())

    # # std of mean of all person
    # df_gender_std_mean_mean = df.copy()
    # df_gender_std_mean_mean = df_gender_std_mean_mean[df_gender_std_mean_mean['gender'] != 'NA']
    # # generate a new df that average the mean of all utterances for each person while maintaining the gender information
    # df_gender_std_mean_mean = df_gender_std_mean_mean.groupby(['subject_id', 'gender']).agg({
    #     'value': mean_f0,
    #     'group_id': 'first'
    #     }).reset_index()
    # # get average of the mean of all person for each gender
    # OA_values = np.array(df_gender_std_mean_mean[df_gender_std_mean_mean['group_id'] == '21'].groupby('gender')['value'].mean().values)
    # PD_values = np.array(df_gender_std_mean_mean[df_gender_std_mean_mean['group_id'] == '22'].groupby('gender')['value'].mean().values)
    # OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    # p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    # reses = ['gender_std_mean_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # # assign the values to the item
    # for k, v in zip(heads, reses):
    #     item[k] = v
    
    # result_datas.append(item.copy())



    return result_datas



def all_level_analysis_utt(df):

  
    item = {}
    heads = ['level', 'OA_values', 'PD_values', 'OA_mean', 'PD_mean', 'OA_std', 'PD_std', 'OA_median', 'PD_median', 'p_ks1', 'p_ks2', 'p_kw', 'p_lev']
    result_datas = []
    
    # utterance level
    df_utt_mean = df.copy()
    OA_values = np.concatenate(df_utt_mean[df_utt_mean['group_id'] == '21']['value'].values)
    # check how many nan in the data
    PD_values = np.concatenate(df_utt_mean[df_utt_mean['group_id'] == '22']['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v
        
    result_datas.append(item.copy())
    

    # utterance level in each gender
    # mean, male
    df_utt_mean = df.copy()
    df_male_OA = df_utt_mean[(df_utt_mean['group_id'] == '21') & (df_utt_mean['gender'] == 'M')]
    df_male_PD = df_utt_mean[(df_utt_mean['group_id'] == '22') & (df_utt_mean['gender'] == 'M')]
    OA_values = np.concatenate(df_male_OA['value'].values)
    PD_values = np.concatenate(df_male_PD['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance_mean_male', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # mean, female
    df_female_OA = df_utt_mean[(df_utt_mean['group_id'] == '21') & (df_utt_mean['gender'] == 'V')]
    df_female_PD = df_utt_mean[(df_utt_mean['group_id'] == '22') & (df_utt_mean['gender'] == 'V')]
    OA_values = np.concatenate(df_female_OA['value'].values)
    PD_values = np.concatenate(df_female_PD['value'].values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['utterance_mean_female', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())
    
    # person level
    # mean of all utterances
    df_person_mean_mean = df.copy()
    OA_values = np.concatenate(df_person_mean_mean[df_person_mean_mean['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.concatenate(df_person_mean_mean[df_person_mean_mean['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['person_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # std of of all utterances
    df_person_std_mean = df.copy()
    # print((df_person_std_mean[df_person_std_mean['group_id'] == '21'].groupby('subject_id')['value'].std().values))
    OA_values = np.array(df_person_std_mean[df_person_std_mean['group_id'] == '21'].groupby('subject_id')['value'].std().values)
    PD_values = np.array(df_person_std_mean[df_person_std_mean['group_id'] == '22'].groupby('subject_id')['value'].std().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['person_std', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # done person level in two gender
    # mean of mean of all male utterances
    df_male_mean_std = df.copy()
    df_male_mean_std = df_male_mean_std[df_male_mean_std['gender'] == 'M']
    OA_values = np.concatenate(df_male_mean_std[df_male_mean_std['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.concatenate(df_male_mean_std[df_male_mean_std['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['male_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # mean of mean of all female utterances
    df_female_mean_std = df.copy()
    df_female_mean_std = df_female_mean_std[df_female_mean_std['gender'] == 'V']
    OA_values = np.concatenate(df_female_mean_std[df_female_mean_std['group_id'] == '21'].groupby('subject_id')['value'].mean().values)
    PD_values = np.concatenate(df_female_mean_std[df_female_mean_std['group_id'] == '22'].groupby('subject_id')['value'].mean().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['female_mean', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # mean of std of all male utterances
    df_male_mean_std = df.copy()
    df_male_mean_std = df_male_mean_std[df_male_mean_std['gender'] == 'M']
    OA_values = np.array(df_male_mean_std[df_male_mean_std['group_id'] == '21'].groupby('subject_id')['value'].std().values)
    PD_values = np.array(df_male_mean_std[df_male_mean_std['group_id'] == '22'].groupby('subject_id')['value'].std().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['male_std', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())

    # mean of std of all female utterances
    df_female_mean_std = df.copy()
    df_female_mean_std = df_female_mean_std[df_female_mean_std['gender'] == 'V']
    OA_values = np.array(df_female_mean_std[df_female_mean_std['group_id'] == '21'].groupby('subject_id')['value'].std().values)
    PD_values = np.array(df_female_mean_std[df_female_mean_std['group_id'] == '22'].groupby('subject_id')['value'].std().values)
    OA_stats, PD_stats = basic_stats(OA_values, PD_values, groupnames=['OA', 'PD'])
    p_ks1, p_ks2, p_kw, p_lev = stats_test(OA_values, PD_values, groupnames=['OA', 'PD'])
    reses = ['female_std', len(OA_values), len(PD_values), OA_stats[0], PD_stats[0], OA_stats[1], PD_stats[1], OA_stats[2], PD_stats[2], p_ks1, p_ks2, p_kw, p_lev]
    # assign the values to the item
    for k, v in zip(heads, reses):
        item[k] = v

    result_datas.append(item.copy())


    return result_datas





def basic_analysis(metadata, featname='shimmer', level='utt', norm=True, log_feat=False):
    

    df = pd.DataFrame(metadata)
    # delete all item in dataframe with group id equal to 11
    df = df[df['group_id'] != '11']
    # import pdb; pdb.set_trace()
    edulist = [2124, 2103, 2108, 2128, 2120, 2114, 2130, 2132]
    edulist = [str(edu) for edu in edulist]
    # remove subject in the education list
    # df = df[~df['subject_id'].isin(edulist)]
    
    
    if level == 'frame':
        # res_df_allexp = pd.DataFrame(all_level_analysis_frame(df))
        res_df_allexp = pd.DataFrame(all_level_analysis_frame(df))
        print(res_df_allexp)
        res_df_PictureNaming = pd.DataFrame(all_level_analysis_frame(df[df['experiment'] == 'exp_1_PictureNaming']))
        res_df_EarlyLate = pd.DataFrame(all_level_analysis_frame(df[df['experiment'] == 'exp_2_EarlyLate']))
        res_df_BoundaryTone = pd.DataFrame(all_level_analysis_frame(df[df['experiment'] == 'exp_3_BoundaryTone']))



    elif level == 'utt':
        res_df_allexp = pd.DataFrame(all_level_analysis_utt(df))
        print(res_df_allexp)
        res_df_PictureNaming = pd.DataFrame(all_level_analysis_utt(df[df['experiment'] == 'exp_1_PictureNaming']))
        res_df_EarlyLate = pd.DataFrame(all_level_analysis_utt(df[df['experiment'] == 'exp_2_EarlyLate']))
        res_df_BoundaryTone = pd.DataFrame(all_level_analysis_utt(df[df['experiment'] == 'exp_3_BoundaryTone']))



    # write all these res to one excel file and one sheet for each experiment
    logornot = 'log_' if log_feat else ''
    normornot = 'unnorm_' if norm is False else ''
    with pd.ExcelWriter('excels/all_level_analysis_' + level + '_' + normornot + logornot + featname + '.xlsx') as writer:
        res_df_allexp.to_excel(writer, sheet_name='all_exp')
        res_df_PictureNaming.to_excel(writer, sheet_name='PictureNaming')
        res_df_EarlyLate.to_excel(writer, sheet_name='EarlyLate')
        res_df_BoundaryTone.to_excel(writer, sheet_name='BoundaryTone')


        

if __name__ == '__main__':
    
    base_folder_path = Path('/data/storage500/Turntaking/wavs_single_channel_normalized_nosil') if norm else Path('/data/storage025/Turntaking/wavs_single_channel_nosil')

    featname = 'shimmer'

    level = 'utt'
    feats2level = {
        'jitter': 'utt',
        'shimmer': 'utt',
        'rp': 'utt',
        'f0': 'frame',
        'energy': 'frame'
    }

    np.set_printoptions(precision=2)
    allfeats = ['jitter', 'shimmer', 'rp', 'f0', 'energy']
    # for feat in allfeats:
    

    #metadata = load_feat(base_folder_path, feature_name=featname)
    load_feat(base_folder_path, feature_name='contrast', threeD=True)
    # basic_analysis(metadata, featname=featname, level=feats2level[featname], norm=False)
    