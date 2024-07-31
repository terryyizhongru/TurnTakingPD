
# extract features via librosa
import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sr=44100):
        self.sr = sr

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # # extract mfcc
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # # extract mel
        # mel = librosa.feature.melspectrogram(y=y, sr=sr)

        # # extract contrast
        # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # # extract spectral centroid
        # spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        # # extract spectral bandwidth
        # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # # extract spectral rolloff
        # spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
        # spec_rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)

        # extract pitch(f0) from time series
        f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                     fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C7'),
                                                     fill_na=0.0)
        f0 = f0[np.newaxis, :]
        voiced_flag = voiced_flag[np.newaxis, :]
        voiced_probs = voiced_probs[np.newaxis, :]


        features = np.concatenate((f0, voiced_flag, voiced_probs), axis=0)
        
        print(features.shape)
        return features

        # # extract zero crossing rate
        # zcr = librosa.feature.zero_crossing_rate(y=y)

        # # extract flatness
        # flatness = librosa.feature.spectral_flatness(y=y)
        
        # # concatenate all features
        # features = np.concatenate((mfcc, mel, contrast, spec_cent, spec_bw, spec_rolloff, spec_rolloff_min, f0, voiced_flag, voiced_probs, zcr, flatness), axis=0)
        # # features = np.concatenate((f0, voiced_probs), axis=0)
        # # Aggregate features
        # # features = np.nan_to_num(features, nan=0.0)
        # # features[~np.isfinite(features)] = 0
        # features = np.vstack((np.mean(features, axis=1))).flatten()
        # print(features.shape)
        
        # return features
    
import os
import re
from pprint import pprint
from tqdm import tqdm
import glob
import multiprocess
from pathlib import Path

base_folder_path = Path('/data/storage025/wavs_single_channel')

# features_folder_path = f'{base_folder_path}-features'
# if not os.path.exists(features_folder_path):
#     os.makedirs(features_folder_path)

# 3 sublists for YA OA PD
BD = [[], [], []]
EL = [[], [], []]
PN = [[], [], []]
all3 = [[], [], []]

def process_file(args):
    wav_file, extractor = args
    dir_path, filename = os.path.split(wav_file)
    # _, dir_name = os.path.split(dir_path)
    
    try:
        #get group id from filename
        group_id = filename.split('_')[0][-4:-2]

        feature = extractor.extract_features(audio_path=wav_file)
        if group_id not in ['11', '21', '22']:
            raise ValueError(f"Invalid group id {group_id}")
        
        features_folder_path = f'{dir_path}-features-pyin'
        if not os.path.exists(features_folder_path):
            os.makedirs(features_folder_path)

        feature_file_path = os.path.join(features_folder_path, os.path.basename(wav_file) + '.npy')
        np.save(feature_file_path, feature)  

        return group_id, feature, f'Processed {wav_file}'
    except Exception as e:
        return None, None, f"Error processing {wav_file}: {e}"
    

def add2list(group_id, feature, ls):
    if group_id == '11':
        ls[0].append(feature)
    elif group_id == '21':
        ls[1].append(feature)
    elif group_id == '22':
        ls[2].append(feature)
    else:
        print(f'Invalid group id {group_id}')
    

feature_extractor = FeatureExtractor()

# load wav files in BoundaryTone  EarlyLate  PictureNaming folders separately
for folder in ['BoundaryTone', 'EarlyLate', 'PictureNaming']:
# for folder in ['test']:

    folder_path = base_folder_path / folder
    wav_files = list(folder_path.glob('*.wav'))
    print(f'Processing {folder} folder...')
    print(f'Found {len(wav_files)} wav files')

    if folder == 'test':
        args = [(wav_file, feature_extractor) for wav_file in wav_files]
        with multiprocess.Pool(1) as pool:
            results = list(tqdm(pool.imap(process_file, args), total=len(wav_files)))
            for group_id, feature, message in results:
                if feature is not None:
                    add2list(group_id, feature, all3)
                else:
                    print(message)
        for sublist in all3:
            print(len(sublist))
    

    if folder == 'BoundaryTone':
        args = [(wav_file, feature_extractor) for wav_file in wav_files]
        with multiprocess.Pool() as pool:
            results = list(tqdm(pool.imap(process_file, args), total=len(wav_files)))
            for group_id, feature, message in results:
                if feature is not None:
                    add2list(group_id, feature, BD)            
                else:
                    print(message)
        # for wav_file in tqdm(wav_files):
        #     process_file(wav_file, feature_extractor, BD)
        for sublist in BD:
            print(len(sublist))
    elif folder == 'EarlyLate':
        args = [(wav_file, feature_extractor) for wav_file in wav_files]
        with multiprocess.Pool() as pool:
            results = list(tqdm(pool.imap(process_file, args), total=len(wav_files)))
            for group_id, feature, message in results:
                if feature is not None:
                    add2list(group_id, feature, EL)
                else:
                    print(message)
        # for wav_file in tqdm(wav_files):
        #     process_file(wav_file, feature_extractor, EL)
        for sublist in EL:
            print(len(sublist))
    elif folder == 'PictureNaming':
        args = [(wav_file, feature_extractor) for wav_file in wav_files]
        with multiprocess.Pool() as pool:
            results = list(tqdm(pool.imap(process_file, args), total=len(wav_files)))
            for group_id, feature, message in results:
                if feature is not None:
                    add2list(group_id, feature, PN)
                else:
                    print(message)
        # for wav_file in tqdm(wav_files):
        #     process_file(wav_file, feature_extractor, PN)
        for sublist in PN:
            print(len(sublist))

    #merge 3lists to all3
    for i in range(3):
        all3[i] += BD[i] + EL[i] + PN[i]
    for sublist in all3:
            print(len(sublist))



