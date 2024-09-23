
import os
import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pdb



base_folder_path = Path('/data/storage025/wavs_single_channel_normalized_nosil/')

# np.set_printoptions(threshold=np.inf)




def get_group_id(filename):
    filename = os.path.basename(filename)
    group_id = filename.split('_')[0][-4:-2]
    if group_id not in ['11', '21', '22']:
        raise ValueError(f"Invalid group id {group_id}")
    return group_id

def add2list(group_id, feature, ls):
    if group_id == '11':
        ls[0].append(feature)
    elif group_id == '21':
        ls[1].append(feature)
    elif group_id == '22':
        ls[2].append(feature)
    else:
        print(f'Invalid group id {group_id}')
        
       
       
def load_f0():
    feature_name = 'f0'
    
    # 3 sublists for YA OA PD
    exp2lists = {'BoundaryTone': [[], [], []], 'EarlyLate': [[], [], []], 'PictureNaming': [[], [], []]}
    avg_diff = []
    # f0_diff_list = open('f0_diff_list.txt', 'w')
    # non_f0_list = open('non_f0_list.txt', 'w')
    for folder in ['BoundaryTone', 'EarlyLate', 'PictureNaming']:
    # for folder in ['BoundaryTone']:
        feature_folder = os.path.join(base_folder_path, folder + '-features', feature_name)
        feature_folder = Path(feature_folder)
        npy_files = list(feature_folder.glob('*.npy'))
        print(f'Processing {folder} folder...')
        print(f'Found {len(npy_files)} npy files')
        

        cnt = 0
        cnt2 = 0
        for npy_file in npy_files:
            feature = np.load(npy_file)
            feature_nonorm = np.load(str(npy_file).replace('_normalized_', '_'))
            avg_diff_each = np.average(np.abs(feature - feature_nonorm))
            # if avg_diff_each > 1:
            #     f0_diff_list.write(f'{os.path.basename(npy_file)}\t{avg_diff_each}\n') 
            avg_diff.append(avg_diff_each)
            
            # check if all 0 value
            if np.max(feature) == 0 and np.min(feature) == 0:
                cnt += 1
                # non_f0_list.write(f'{os.path.basename(npy_file)}\n')
                continue
            
            if np.max(feature) < 60:
                print(f'max value < 60 in {npy_file}')
                cnt2 += 1
                # non_f0_list.write(f'{os.path.basename(npy_file)}\n')

                continue
            
            group_id = get_group_id(npy_file.stem)
            add2list(group_id, feature, exp2lists[folder])
        print(f'{cnt} files with all 0 values')
        print(f'{cnt2} files with max value < 60Hz')
        
    
    avg_diff = np.array(avg_diff)
    # print top 10 largest diff
    print(len(avg_diff))
    # print(np.sort(avg_diff)[-800])
    print('mean diff: ', np.min(avg_diff))
    print('avg diff: ', np.average(avg_diff)) 
    # close
    # f0_diff_list.close()
    # non_f0_list.close()
    
    all3 = [[], [], []] 
    for i in range(3):
        all3[i] += exp2lists['BoundaryTone'][i] + exp2lists['EarlyLate'][i] + exp2lists['PictureNaming'][i]

    # deep copy BD
    all3_final = [[], [], []]

    for i in range(3):
        tmp = np.concatenate(all3[i], axis=0)
        # remove all 0 value
        # print(tmp.shape)
        tmp = tmp[tmp != 0]    
        # print(tmp.shape)
        # remove value > 500
        tmp = tmp[tmp < 500.0]
        # print(tmp.shape)
        all3_final[i] = tmp
        
        tmp = np.concatenate(exp2lists['BoundaryTone'][i])
        tmp = tmp[tmp != 0]
        tmp = tmp[tmp < 500.0]
        exp2lists['BoundaryTone'][i] = tmp
        
        tmp = np.concatenate(exp2lists['EarlyLate'][i])
        tmp = tmp[tmp != 0]
        tmp = tmp[tmp < 500.0]
        exp2lists['EarlyLate'][i] = tmp

        tmp = np.concatenate(exp2lists['PictureNaming'][i]) if len(exp2lists['PictureNaming'][i]) > 0 else np.array([0])
        tmp = tmp[tmp != 0]
        tmp = tmp[tmp < 500.0]
        exp2lists['PictureNaming'][i] = tmp
        
    return all3_final, exp2lists


def load_frame_feat(feature_name='energy'):
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
            feature_nonorm = np.load(str(npy_file).replace('_normalized_', '_'))
            avg_diff_each = np.average(np.abs(feature - feature_nonorm))
            # if avg_diff_each > 1:
            #     f0_diff_list.write(f'{os.path.basename(npy_file)}\t{avg_diff_each}\n') 
            avg_diff.append(avg_diff_each)
            
            # check if all 0 value
            if np.max(feature) == 0 and np.min(feature) == 0:
                cnt += 1
                # non_f0_list.write(f'{os.path.basename(npy_file)}\n')
                continue
            
            group_id = get_group_id(npy_file.stem)
            add2list(group_id, feature, exp2lists[folder])
        print(f'{cnt} files with all 0 values')
        
    
    avg_diff = np.array(avg_diff)
    # print top 10 largest diff
    print(len(avg_diff))
    print('top10 max diff: ', np.sort(avg_diff)[-10:])
    print('mean diff: ', np.min(avg_diff))
    print('avg diff: ', np.average(avg_diff)) 
    # close
    # f0_diff_list.close()
    # non_f0_list.close()
    
    all3 = [[], [], []] 
    for i in range(3):
        all3[i] += exp2lists['BoundaryTone'][i] + exp2lists['EarlyLate'][i] + exp2lists['PictureNaming'][i]

    # deep copy BD
    all3_final = [[], [], []]

    for i in range(3):
        tmp = np.concatenate(all3[i], axis=0)
        # remove all 0 value
        print(tmp.shape)
        tmp = tmp[tmp != 0]    
        print(tmp.shape)
        # remove value > 500
        # tmp = tmp[tmp < 500.0]
        # print(tmp.shape)
        all3_final[i] = tmp
        
        tmp = np.concatenate(exp2lists['BoundaryTone'][i])
        tmp = tmp[tmp != 0]
        exp2lists['BoundaryTone'][i] = tmp

        tmp = np.concatenate(exp2lists['EarlyLate'][i])
        tmp = tmp[tmp != 0]
        exp2lists['EarlyLate'][i] = tmp

        tmp = np.concatenate(exp2lists['PictureNaming'][i]) if len(exp2lists['PictureNaming'][i]) > 0 else np.array([0])
        tmp = tmp[tmp != 0]
        exp2lists['PictureNaming'][i] = tmp
        
    return all3_final, exp2lists
    

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
                add2list(group_id, rp, exp2lists[folder])
                cnt += 1
                
        print(f'Processing {feature_list} ...')
        print(f'Found {cnt}')


    all3 = [[], [], []] 
    for i in range(3):
        all3[i] += exp2lists['BoundaryTone'][i] + exp2lists['EarlyLate'][i] + exp2lists['PictureNaming'][i]
    
        # deep copy BD
    all3_final = [[], [], []]

    for i in range(3):
        tmp = np.array(all3[i])
        # remove all 0 value
        print(tmp.shape)
        tmp = tmp[tmp != 0]    
        print(tmp.shape)
        # remove value > 500
        # tmp = tmp[tmp < 500.0]
        # print(tmp.shape)
        all3_final[i] = tmp
        
        tmp = np.array(exp2lists['BoundaryTone'][i])
        tmp = tmp[tmp != 0]
        exp2lists['BoundaryTone'][i] = tmp

        tmp = np.array(exp2lists['EarlyLate'][i])
        tmp = tmp[tmp != 0]
        exp2lists['EarlyLate'][i] = tmp

        tmp = np.array(exp2lists['PictureNaming'][i]) if len(exp2lists['PictureNaming'][i]) > 0 else np.array([0])
        tmp = tmp[tmp != 0]
        exp2lists['PictureNaming'][i] = tmp
    
    return all3_final, exp2lists




from scipy.stats import shapiro, kruskal, kstest, norm, levene

def stats_test(all3):
    groups = ['YA', 'OA', 'PD']
    for i, group in enumerate(all3):
        print(f"Group {groups[i]} - Mean: {np.mean(group):.2f}, Std: {np.std(group):.2f}, Median: {np.median(group):.2f}")
        # print size of each group
        print(f"Group {groups[i]} - Size: {len(group)}")

    for i, group in enumerate(all3):
        # data_mean = np.mean(group)
        # data_std = np.std(group)
        # standardized_data = (group - data_mean) / data_std
        d_stat, p_value = kstest(group, 'norm')
        print(f"kstest p-value: {p_value}")
        # stat, p = shapiro(group)
        # print(f"Group {groups[i]} - Shapiro-Wilk p-value: {p}")

    # print shape of all3
    print(all3[0].shape, all3[1].shape, all3[2].shape) 
    
    stat, p = kruskal(all3[0], all3[1])
    print(f"Kruskal-Wallis test p value between group {groups[0]} and {groups[1]} : {p}")

  
    stat, p = kruskal(all3[1], all3[2])
    print(f"Kruskal-Wallis test p value between group {groups[1]} and {groups[2]} : {p}")

    levene_stat, levene_p = levene(all3[0], all3[1], center='median')
    print(f"Levene test p value between group {groups[0]} and {groups[1]} : {levene_p}")


    levene_stat, levene_p = levene(all3[1], all3[2], center='median')
    print(f"Levene test p value between group {groups[1]} and {groups[2]} : {levene_p}")
    

    
def stats_test_exp(all3):
    groups = ['YA', 'OA', 'PD']

    for i, group in enumerate(all3):
        print(f"Group {i} - Mean: {np.mean(group):.2f}, Std: {np.std(group):.2f}, Median: {np.median(group):.2f}")


    for i, group in enumerate(all3):
        data_mean = np.mean(group)
        data_std = np.std(group)
        standardized_data = (group - data_mean) / data_std
        d_stat, p_value = kstest(standardized_data, 'norm')
        print(f"p-value: {p_value}")
        # stat, p = shapiro(group)
        # print(f"Group {groups[i]} - Shapiro-Wilk p-value: {p}")


    
    stat, p = kruskal(all3[0], all3[1])
    levene_stat, levene_p = levene(all3[0], all3[1])
    print(f"Kruskal-Wallis test p value between group {groups[0]} and {groups[1]} : {p}")
    print(f"Levene test p value between group {groups[0]} and {groups[1]} : {levene_p}")
    stat, p = kruskal(all3[1], all3[2])
    levene_stat, levene_p = levene(all3[1], all3[2])
    print(f"Kruskal-Wallis test p value between group {groups[1]} and {groups[2]} : {p}")
    print(f"Levene test p value between group {groups[1]} and {groups[2]} : {levene_p}")
    

    

# print('================f0================')
# stats_test(all3_f0)
# print('================energy================')
# stats_test(all3_energy)
# print('================rp================')
# stats_test(all3_rp)




all3_f0, exp2list_f0 = load_f0()
# all3_energy, exp2list_energy = load_frame_feat(feature_name='energy')
# all3_rp, exp2list_rp = load_rp()
# print(exp2list)


for exp in ['EarlyLate', 'BoundaryTone', 'PictureNaming']:
    print(f'================{exp}================')
    # print('================f0================')
    # stats_test(exp2list_f0[exp])
    # print('================energy================')
    # stats_test(exp2list_energy[exp])
    # print('================rp================')
    # stats_test(exp2list_rp[exp])


        

# print('================f0================')
# stats_test(all3_f0)
# print('================energy================')
# stats_test(all3_energy)
# print('================rp================')
# stats_test(all3_rp)
