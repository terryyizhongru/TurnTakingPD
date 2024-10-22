
# extract features via librosa
import os
import sys
from tqdm import tqdm
import multiprocess
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

        
class FeatureExtractor:
    def __init__(self, sr=44100, frame_length=4096):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = self.frame_length // 4



    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # # extract mfcc
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # # extract mel
        # mel = librosa.feature.melspectrogram(y=y, sr=sr)

        # extract contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # extract spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        # extract spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # extract spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
        spec_rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)

        # # extract pitch(f0) from time series
        # f0, voiced_flag, voiced_probs = librosa.pyin(y,
        #                                              fmin=librosa.note_to_hz('C2'),
        #                                              fmax=librosa.note_to_hz('C7'),
        #                                              fill_na=0.0)
        # f0 = f0[np.newaxis, :]
        # voiced_flag = voiced_flag[np.newaxis, :]
        # voiced_probs = voiced_probs[np.newaxis, :]

    
        # extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)

        # extract flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        
        praatsound = parselmouth.Sound(str(audio_path))
        # concatenate all features
        # features = np.concatenate((mfcc, mel, contrast, spec_cent, spec_bw, spec_rolloff, spec_rolloff_min, f0, voiced_flag, voiced_probs, zcr, flatness), axis=0)
        # # features = np.concatenate((f0, voiced_probs), axis=0)
        # # Aggregate features
        # # features = np.nan_to_num(features, nan=0.0)
        # # features[~np.isfinite(features)] = 0
        # features = np.vstack((np.mean(features, axis=1))).flatten()
        # print(features.shape)
        
        # return features
        formant = praatsound.to_formant_burg(time_step=0.01, max_number_of_formants=3, maximum_formant=5500)
    
        # 获取时间轴
        times = formant.xs()
        
        # 初始化列表
        F1 = []
        F2 = []
        F3 = []
        
        # 提取每个时间点的Formants
        for t in times:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            
            # 处理未定义的Formants
            F1.append(f1 if f1 != 0 else np.nan)
            F2.append(f2 if f2 != 0 else np.nan)
            F3.append(f3 if f3 != 0 else np.nan)
        # change to numpy array
        F1, F2, F3 = np.array(F1), np.array(F2), np.array(F3)
        harmonicity = call(praatsound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_value = call(harmonicity, "Get mean", 0, 0)
        
        featurelist = ['contrast', 'spec_cent', 'spec_bw', 'spec_rolloff', 'spec_rolloff_min', 'zcr', 'flatness', 'F1', 'F2', 'F3', 'hnr_value']
        features = (contrast, spec_cent, spec_bw, spec_rolloff, spec_rolloff_min, zcr, flatness, F1, F2, F3, hnr_value)
        for feat in features:
            if type(feat) == float:
                continue
                # print(feat)
            elif type(feat) == np.ndarray:
                if feat.shape[0] == 1 and len(feat.shape) == 2:
                    feat = feat[0]
                # print(feat.shape)
    
        return features, featurelist

    def extract_jitter_shimmer(self, praatsound):
        sound = praatsound
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        
        jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        print(jitter_local, shimmer_local)
        return jitter_local, shimmer_local


    
    # def get_vad(self, y, sr, mode=2):
    #     vad = webrtcvad.Vad()
    #     vad.set_mode(mode)

    #     sample_rate = 16000
    #     # downsampling from 44100 to 16000 and framing
    #     y_16k = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
    #     # transfer to bytes
    #     y_16k = np.int16(y_16k * 32768).tobytes()
    #     # calculate bytes of 10ms in 16000Hz
    #     step = int(sample_rate * 0.01) * 2

    #     # assert error if len(y_16k) < step
    #     assert len(y_16k) >= step

    #     # loop the bytes in 10ms frames
    #     frames = []
    #     for i in range(0, len(y_16k) - step, step):
    #         frames.append(y_16k[i:i+step])

    #     vadres = []
    #     for i, frame in enumerate(frames):
    #         vadres.append(vad.is_speech(frame, sample_rate))
    #     return vadres
    
    def process_file(self, args):
        wav_file, extractor = args
        dir_path, filename = os.path.split(wav_file)
        # _, dir_name = os.path.split(dir_path)
        
        try:
            #get group id from filename
            group_id = filename.split('_')[0][-4:-2]
            if group_id not in ['11', '21', '22']:
                raise ValueError(f"Invalid group id {group_id}")
            
            features, featurelist = extractor.extract_features(audio_path=wav_file)
            features_folder_path = f'{dir_path}-features'
            if not os.path.exists(features_folder_path):
                os.makedirs(features_folder_path)

            for f in featurelist:
                feature_subfolder_path = os.path.join(features_folder_path, f)
                if not os.path.exists(feature_subfolder_path):
                    os.makedirs(feature_subfolder_path)
                    
                feature_file_path = os.path.join(feature_subfolder_path, os.path.basename(wav_file) + '_' + f + '.npy')
                np.save(feature_file_path, features[featurelist.index(f)])

            return group_id, features, f'Processed {wav_file}, extracted features: {str(featurelist)}'
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
    

def multiprocess_extract():

    feature_extractor = FeatureExtractor()

    # load wav files in BoundaryTone  EarlyLate  PictureNaming folders separately
    for folder in ['BoundaryTone', 'EarlyLate', 'PictureNaming']:
    # for folder in ['PictureNaming']:
        folder_path = base_folder_path / folder
        wav_files = list(folder_path.glob('*.wav'))
        print(f'Processing {folder} folder...')
        print(f'Found {len(wav_files)} wav files')


        if folder == 'BoundaryTone':
            args = [(wav_file, feature_extractor) for wav_file in wav_files]
            with multiprocess.Pool() as pool:
                results = list(tqdm(pool.imap(feature_extractor.process_file, args), total=len(wav_files)))
                for group_id, features, message in results:
                    if features is not None:
                        add2list(group_id, features, BD)            
                        print(message)
                    else:
                        print(message)
            # for wav_file in tqdm(wav_files):
            #     process_file(wav_file, feature_extractor, BD)
            for sublist in BD:
                print(len(sublist))
                
        elif folder == 'EarlyLate':
            args = [(wav_file, feature_extractor) for wav_file in wav_files]
            with multiprocess.Pool() as pool:
                results = list(tqdm(pool.imap(feature_extractor.process_file, args), total=len(wav_files)))
                for group_id, features, message in results:
                    if features is not None:
                        add2list(group_id, features, EL)
                        print(message)
                    else:
                        print(message)
            # for wav_file in tqdm(wav_files):
            #     process_file(wav_file, feature_extractor, EL)
            for sublist in EL:
                print(len(sublist))
                
        elif folder == 'PictureNaming':
            args = [(wav_file, feature_extractor) for wav_file in wav_files]
            with multiprocess.Pool() as pool:
                results = list(tqdm(pool.imap(feature_extractor.process_file, args), total=len(wav_files)))
                for group_id, features, message in results:
                    if features is not None:
                        add2list(group_id, features, PN)
                        print(message)
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

def singleprocess_extract():
    feature_extractor = FeatureExtractor()

    # load wav files in BoundaryTone  EarlyLate  PictureNaming folders separately
    # for folder in ['BoundaryTone', 'EarlyLate', 'PictureNaming']:
    for folder in ['PictureNaming']:
        folder_path = base_folder_path / folder
        wav_files = list(folder_path.glob('*.wav'))
        print(f'Processing {folder} folder...')
        print(f'Found {len(wav_files)} wav files')

        results = []
        for wav_file in tqdm(wav_files[:100]):
            args = (wav_file, feature_extractor)
            results.append(feature_extractor.process_file(args))
        for group_id, features, message in results:
            if features is not None:
                add2list(group_id, features, PN)
                print(message)
            else:
                print(message)

    

if len(sys.argv) != 2:
    print("Usage: python save_features.py [base_folder_path]")
    sys.exit(1)
base_folder_path = Path(sys.argv[1])

# features_folder_path = f'{base_folder_path}-features'
# if not os.path.exists(features_folder_path):
#     os.makedirs(features_folder_path)

# 3 sublists for YA OA PD
BD = [[], [], []]
EL = [[], [], []]
PN = [[], [], []]
all3 = [[], [], []]

multiprocess_extract()