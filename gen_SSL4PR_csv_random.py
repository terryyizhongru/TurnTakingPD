import os
import csv
import random
import math
from collections import defaultdict
import sys
import numpy as np  # 如果使用了 numpy.random

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

def split_wavs_10fold_balanced(example_wavs, num_folds=10):
    """
    Splits WAVs into 10 folds, trying to balance speakers whose IDs start with '21' and '22'.
    Also includes any other speakers in a separate group.
    Returns a list of (train_wavs, test_wavs) for each fold.
    """
    random.seed(42)
    # Group WAV paths by speaker
    speaker_dict = defaultdict(list)
    for wav in example_wavs:
        sid = get_speaker_id(wav)
        speaker_dict[sid].append(wav)

    # Partition speakers by prefix
    speakers_21, speakers_22, speakers_others = partition_speakers_by_prefix(list(speaker_dict.keys()))

    # Create folds independently for each group
    folds_21 = create_folds_for_speakers(speakers_21, num_folds=num_folds)
    folds_22 = create_folds_for_speakers(speakers_22, num_folds=num_folds)
    folds_others = create_folds_for_speakers(speakers_others, num_folds=num_folds)

    # Combine folds from each group index
    folds = []
    for i in range(num_folds):
        test_speakers_21 = folds_21[i] if i < len(folds_21) else []
        test_speakers_22 = folds_22[i] if i < len(folds_22) else []
        test_speakers_others = folds_others[i] if i < len(folds_others) else []
        test_speakers = test_speakers_21 + test_speakers_22 + test_speakers_others

        # Train speakers are everything not in the test fold
        train_speakers = (
            set(speaker_dict.keys())
            - set(test_speakers)
        )

        # Convert speaker IDs to WAV paths
        test_wavs = []
        for sp in test_speakers:
            test_wavs.extend(speaker_dict[sp])
        train_wavs = []
        for sp in train_speakers:
            train_wavs.extend(speaker_dict[sp])

        folds.append((train_wavs, test_wavs))

    return folds

# def split_wavs_10fold(example_wavs, num_folds=10):
#     """
#     Splits wavs into 10 folds based on speaker ID, ensuring speaker separation
#     between train and test sets.
#     Returns a list of tuples (train_wavs, test_wavs) for each fold.
#     """

#     # Group wavs by speaker
#     speaker_dict = {}
#     for wav in example_wavs:
#         sid = get_speaker_id(wav)
#         speaker_dict.setdefault(sid, []).append(wav)

#     # Shuffle speaker IDs
#     all_speakers = list(speaker_dict.keys())
#     random.shuffle(all_speakers)

#     # Create folds of speaker IDs
#     fold_size = math.ceil(len(all_speakers) / num_folds)
#     folds = []
#     for i in range(num_folds):
#         test_speakers = all_speakers[i * fold_size : (i + 1) * fold_size]
#         train_speakers = [s for s in all_speakers if s not in test_speakers]

#         # Collect wav paths
#         test_wavs = []
#         for s in test_speakers:
#             test_wavs.extend(speaker_dict[s])

#         train_wavs = []
#         for s in train_speakers:
#             train_wavs.extend(speaker_dict[s])

#         folds.append((train_wavs, test_wavs))

#     return folds

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
            demogr[pid] = (age, sex)
    return demogr

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
            sex = "F" if sex == "V" else "M"
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