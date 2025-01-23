import os
import csv
import random
import math
from collections import defaultdict
import sys
import numpy as np  # 如果使用了 numpy.random



DEMOGR_FILE = "/home/yzhong/gits/TurnTakingPD/sync_private/demogr_perpp.txt"

def load_demographics(file_path):
    demogr = {}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        header = lines[0].split()
        idx_id = header.index("participantnummer")
        idx_age = header.index("leeftijd")
        idx_sex = header.index("geslacht")
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            pid = parts[idx_id]
            age = parts[idx_age]
            sex = parts[idx_sex]
            sex = sex if sex == "M" else "F"
            demogr[pid] = (age, sex)
    return demogr

def group_speakers_by_status_and_sex(speaker_dict, demogr):
    groups = {
        'hc': {'M': [], 'F': []},
        'pd': {'M': [], 'F': []}
    }
    for speaker_id, wavs in speaker_dict.items():
        status = 'pd' if speaker_id.startswith('22') else 'hc'
        sex = demogr.get(speaker_id, ('', 'M'))[1]  # Default to 'M' if not found
        groups[status][sex].append(speaker_id)
    return groups


def get_speaker_id(wav_path):
    """
    Adjust this logic to correctly extract the speaker ID.
    Example: '.../subj-1104_something.wav' -> '1104'
    """
    base = os.path.basename(wav_path)
    # Assume format like 'subj-1104_something.wav'
    # Split on '_' then take first part, split on '-' and take last
    first_part = base.split("_")[0]      # 'subj-1104'
    speaker_id = first_part.replace("subj-", "")
    return speaker_id

def partition_speakers_by_prefix(all_speakers):
    """
    Partition speaker IDs into buckets based on whether they start with '21', '22', or neither.
    Returns three lists: speakers_21, speakers_22, speakers_others
    """
    speakers_21 = []
    speakers_22 = []
    speakers_others = []
    for sp in all_speakers:
        if sp.startswith("21"):
            speakers_21.append(sp)
        elif sp.startswith("22"):
            speakers_22.append(sp)
        else:
            speakers_others.append(sp)
    return speakers_21, speakers_22, speakers_others

def create_folds_for_speakers(speakers, num_folds=10):
    """
    Given a list of speaker IDs, split them into num_folds folds.
    Returns a list of lists, where each sub-list is the speaker IDs in that fold.
    """
    if not speakers:
        return [[] for _ in range(num_folds)]
    
    random.seed(42)
    random.shuffle(speakers)
    min_fold_size = max(1, len(speakers) // num_folds)  # Ensure at least 1 speaker per fold
    
    folds = []
    remaining = speakers.copy()
    
    # Distribute speakers evenly
    for i in range(num_folds):
        if not remaining:
            folds.append([])
            continue
            
        fold_size = min(min_fold_size, len(remaining))
        fold_speakers = remaining[:fold_size]
        remaining = remaining[fold_size:]
        folds.append(fold_speakers)
        
    # Distribute any remaining speakers
    fold_idx = 0
    while remaining:
        folds[fold_idx].append(remaining.pop(0))
        fold_idx = (fold_idx + 1) % num_folds
        
    return folds

# def create_folds_for_speakers(speakers, num_folds=10):
#     """
#     Given a list of speaker IDs, split them into num_folds folds.
#     Returns a list of lists, where each sub-list is the speaker IDs in that fold.
#     """
#     random.shuffle(speakers)
#     fold_size = math.ceil(len(speakers) / num_folds)
#     folds = []
#     for i in range(num_folds):
#         fold_speakers = speakers[i * fold_size : (i + 1) * fold_size]
#         folds.append(fold_speakers)
#     return folds

# def split_wavs_10fold_balanced(example_wavs, num_folds=10):
#     """
#     Splits WAVs into 10 folds, trying to balance speakers whose IDs start with '21' and '22'.
#     Also includes any other speakers in a separate group.
#     Returns a list of (train_wavs, test_wavs) for each fold.
#     """
#     random.seed(42)
#     # Group WAV paths by speaker
#     speaker_dict = defaultdict(list)
#     for wav in example_wavs:
#         sid = get_speaker_id(wav)
#         speaker_dict[sid].append(wav)

#     # Partition speakers by prefix
#     speakers_21, speakers_22, speakers_others = partition_speakers_by_prefix(list(speaker_dict.keys()))

#     # Create folds independently for each group
#     folds_21 = create_folds_for_speakers(speakers_21, num_folds=num_folds)
#     folds_22 = create_folds_for_speakers(speakers_22, num_folds=num_folds)
#     folds_others = create_folds_for_speakers(speakers_others, num_folds=num_folds)

#     # Combine folds from each group index
#     folds = []
#     for i in range(num_folds):
#         test_speakers_21 = folds_21[i] if i < len(folds_21) else []
#         test_speakers_22 = folds_22[i] if i < len(folds_22) else []
#         test_speakers_others = folds_others[i] if i < len(folds_others) else []
#         test_speakers = test_speakers_21 + test_speakers_22 + test_speakers_others

#         # Train speakers are everything not in the test fold
#         train_speakers = (
#             set(speaker_dict.keys())
#             - set(test_speakers)
#         )

#         # Convert speaker IDs to WAV paths
#         test_wavs = []
#         for sp in test_speakers:
#             test_wavs.extend(speaker_dict[sp])
#         train_wavs = []
#         for sp in train_speakers:
#             train_wavs.extend(speaker_dict[sp])

#         folds.append((train_wavs, test_wavs))

#     return folds
import copy

def split_wavs_10fold_balanced(example_wavs, num_folds=10):
    random.seed(42)
    speaker_dict = defaultdict(list)
    for wav in example_wavs:
        sid = get_speaker_id(wav)
        speaker_dict[sid].append(wav)

    demogr = load_demographics(DEMOGR_FILE)
    original_grouped_speakers = group_speakers_by_status_and_sex(speaker_dict, demogr)
    available_grouped_speakers = copy.deepcopy(original_grouped_speakers)

    folds = []
    add1 = {}
    for fo in range(num_folds):
        print("fold: ", fo+1)
        grouped_speakers = copy.deepcopy(original_grouped_speakers)
        test_speakers = []
        for status in ['hc', 'pd']:
            for sex in ['M', 'F']:
                if len(available_grouped_speakers[status][sex]) >= 2:
                    available_speakers = available_grouped_speakers[status][sex]
                    print(len(available_speakers), status, sex)
                elif len(available_grouped_speakers[status][sex]) == 1:
    
                    available_speakers = available_grouped_speakers[status][sex]
                    print(len(available_speakers), status, sex)
                    while len(add1) == 0 or add1[0] in available_speakers:
                        add1 = random.sample(grouped_speakers[status][sex], 1)
                    available_speakers.extend(add1)
                    add1 = {}
                    print(len(available_speakers), status, sex)
                else:
                    available_speakers = grouped_speakers[status][sex]
                num_to_select = min(2, len(available_speakers))
                selected = random.sample(available_speakers, num_to_select)
                test_speakers.extend(selected)
                
                if len(available_grouped_speakers[status][sex]) >= 1:
                    for sp in selected:
                        available_grouped_speakers[status][sex].remove(sp) 
                        
        train_speakers = [sp for group in grouped_speakers.values() for subgroup in group.values() for sp in subgroup if sp not in test_speakers]

        print([[sp, demogr.get(sp, ('', ''))] for sp in test_speakers])
            
        test_wavs = [wav for sp in test_speakers for wav in speaker_dict[sp]]
        train_wavs = [wav for sp in train_speakers for wav in speaker_dict[sp]]

        folds.append((train_wavs, test_wavs))

    return folds


def print_hc_pd_proportions(csv_file):
    hc_count, pd_count, total = 0, 0, 0
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # relies on generate_csv writing a header row
        for row in reader:
            total += 1
            status = row.get("status", "")
            if status == "hc":
                hc_count += 1
            elif status == "pd":
                pd_count += 1
    if total > 0:
        print(f"{csv_file} total: {total}, hc: {hc_count} ({hc_count/total:.2f}), pd: {pd_count} ({pd_count/total:.2f})")
    else:
        print(f"{csv_file} is empty")

def generate_csv(wav_files, output_csv):
    demogr = load_demographics(DEMOGR_FILE)
    header = [
        "original_id",
        "audio_path",
        "status",
        "speaker_id",
        "group",
        "UPDRS",
        "UPDRS-speech",
        "H/Y",
        "SEX",
        "AGE",
        "time after diagnosis"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for wavpath in wav_files:
            basename = os.path.basename(wavpath)  # e.g. 22-1101_something.wav
            foldername = os.path.basename(os.path.dirname(wavpath))
            original_id = basename.rstrip('.wav') + "_" + foldername
            speaker_id = basename.split("_")[0].split("-")[-1]
            
            if speaker_id.startswith("22"):
                status = "pd"
            else:
                status = "hc"
            if speaker_id in demogr:
                age, sex = demogr[speaker_id]
            else:
                age, sex = "", ""
            sex = "M" if sex == "M" else "F"
            age = "" if age == "NA" else age
            row = [
                original_id,
                wavpath,
                status,
                speaker_id,
                "1",  # group
                "",   # UPDRS
                "",   # UPDRS-speech
                "",   # H/Y
                sex,
                age,
                ""    # placeholder
            ]
            writer.writerow(row)

if __name__ == "__main__":
    # 在所有随机操作之前设置种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    with open(sys.argv[1], "r") as f:
        all_wavs = f.read().splitlines()
    # remove wav files with "subj-11" in the path
    all_wavs = [wav for wav in all_wavs if "subj-11" not in wav]
    all_wavs.sort()
    random.shuffle(all_wavs)

    # split into folds
    folds = split_wavs_10fold_balanced(all_wavs, num_folds=10)

    # create subfolders and generate train.csv / test.csv
    output_folder = "splits/" + sys.argv[1].split('/')[-1] + "_10fold"
    output_folder_unnorm = "splits/" + sys.argv[1].split('/')[-1] + "_unnorm_10fold"
    
    os.makedirs(output_folder, exist_ok=True)
    for i, (train_wavs, test_wavs) in enumerate(folds, start=1):
        fold_dir = os.path.join(output_folder, f"TRAIN_TEST_{i}")
        fold_dir_unnorm = os.path.join(output_folder_unnorm, f"TRAIN_TEST_{i}")
        
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(fold_dir_unnorm, exist_ok=True)
        
        train_csv = os.path.join(fold_dir, "train.csv")
        test_csv = os.path.join(fold_dir, "test.csv")
        train_csv_unnorm = os.path.join(fold_dir_unnorm, "train.csv")
        test_csv_unnorm = os.path.join(fold_dir_unnorm, "test.csv")
        
        generate_csv(train_wavs, train_csv)
        generate_csv(test_wavs, test_csv)
        print_hc_pd_proportions(train_csv)
        print_hc_pd_proportions(test_csv)
        

        if all("new_merged_wavs" in wav for wav in train_wavs):
            train_wavs = [wav.replace("new_merged_wavs", "new_merged_wavs_unnorm") for wav in train_wavs]
            test_wavs = [wav.replace("new_merged_wavs", "new_merged_wavs_unnorm") for wav in test_wavs]
        elif all("all_batch1234" in wav for wav in train_wavs):
            train_wavs = [wav.replace("all_batch1234", "all_batch1234_unnorm") for wav in train_wavs]
            test_wavs = [wav.replace("all_batch1234", "all_batch1234_unnorm") for wav in test_wavs]
            
        generate_csv(train_wavs, train_csv_unnorm)
        generate_csv(test_wavs, test_csv_unnorm)
        
        # print_hc_pd_proportions(train_csv)
        # print_hc_pd_proportions(test_csv)