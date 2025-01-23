import os
import csv
import sys
import math
import statistics
import argparse
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.silence import split_on_silence

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

def measure_audio_features(signal_file):
    audio = AudioSegment.from_file(signal_file)
    duration = len(audio) / 1000.0
    loudness = audio.dBFS
    return duration, loudness

def summary_stats(values):
    # Filter out NaN/inf values
    finite_vals = [v for v in values if math.isfinite(v)]
    if not finite_vals:
        return (0, 0, 0, 0)
    return (
        min(finite_vals),
        max(finite_vals),
        statistics.mean(finite_vals),
        statistics.pstdev(finite_vals)  # population std
    )

def get_hc_pd_stats(train_csv, test_csv, args):
    subjects_data = {}

    def process_row(row):
        # row example: [stimulus, path, label, subject_id, col5, col6, col7, col8, sex, age, ...]
        label = row[2].strip().lower()
        subject_id = row[3].strip()
        sex = row[8].strip().upper() if len(row) > 8 else ''
        age_str = row[9].strip() if len(row) > 9 else ''

        try:
            age = float(age_str)
        except ValueError:
            age = None

        return subject_id, label, sex, age

    csv_files = [train_csv, test_csv] if args.set == "all" else [test_csv]
    for f in csv_files:
        if not os.path.isfile(f):
            continue
        with open(f, 'r', newline='', encoding='utf8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row or 'original_id' in row[0].lower() or len(row) < 4:
                    continue
                subject_id, label, sex, age = process_row(row)
                if not subject_id:
                    continue
                if subject_id not in subjects_data:
                    subjects_data[subject_id] = {
                        'label': label,
                        'sex': sex,
                        'age': age
                    }

    hc_subjects = []
    pd_subjects = []
    for subj, info in subjects_data.items():
        label = info['label']
        if label == 'hc':
            hc_subjects.append(info)
        elif label == 'pd':
            pd_subjects.append(info)

    def compute_stats(subject_list):
        male_count = sum(1 for s in subject_list if s['sex'] == 'M')
        female_count = sum(1 for s in subject_list if s['sex'] == 'F')
        ages = [s['age'] for s in subject_list if s['age'] is not None]

        avg_age = sum(ages) / len(ages) if ages else 0
        age_min = min(ages) if ages else 0
        age_max = max(ages) if ages else 0

        return {
            'count': len(subject_list),
            'male': male_count,
            'female': female_count,
            'avg_age': avg_age,
            'age_min': age_min,
            'age_max': age_max
        }
    # print(f"Total unique subjects: {len(subjects_data)}")
    
    return compute_stats(hc_subjects), compute_stats(pd_subjects)

def main():
    parser = argparse.ArgumentParser(description="Calculate audio stats for train+test or just test.")
    parser.add_argument("folder", help="Folder containing train.csv and test.csv")
    parser.add_argument("--set", choices=["all", "test"], default="all",
                        help="Choose 'all' to aggregate train+test, or 'test' for test only.")
    parser.add_argument("--mode", choices=["all", "one"], default="one",
                    help="Choose 'all' to aggregate train+test, or 'test' for test only.")
    args = parser.parse_args()

    all_test_subjects = set()
    demo_dict = load_demographics(DEMOGR_FILE)

    if args.mode == "all":
        folder = args.folder
        subjects_list = []
        for subfolder in os.listdir(folder):
            train_csv = os.path.join(folder, subfolder, 'train.csv')
            test_csv = os.path.join(folder, subfolder, 'test.csv')
            # print(test_csv)
            
            hc_stats, pd_stats = get_hc_pd_stats(train_csv, test_csv, args)
            if os.path.isfile(test_csv):
                with open(test_csv, 'r', newline='', encoding='utf8') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if not row or 'original_id' in row[0].lower() or len(row) < 4:
                            continue
                        subject_id = row[3].strip()
                        if subject_id:
                            all_test_subjects.add(subject_id)
                            subjects_list.append(subject_id)
                            
        print(len(subjects_list))
    
        # calculate the proportion of gender of subjects list
        genderlist = [demo_dict.get(subject, ('', ''))[1] for subject in subjects_list]
        print("Female proportion:", genderlist.count('V')/len(genderlist))
        
        print(f"Total unique subjects in all test.csv files: {len(all_test_subjects)}")

    if args.mode == "one":
        folder = args.folder
        train_csv = os.path.join(folder, 'train.csv')
        test_csv = os.path.join(folder, 'test.csv')
        
        hc_stats, pd_stats = get_hc_pd_stats(train_csv, test_csv, args)
        
        # Collect stats for each audio file
        durations = []
        loudnesses = []

        # Track audio count per subject
        subject2files = {}
        subject2duration = {}

        # Decide which files to read based on mode
        csv_files = []
        if args.set == "all":
            csv_files = [train_csv, test_csv]
        else:  # "test"
            csv_files = [test_csv]

        for f in csv_files:
            if not os.path.isfile(f):
                continue
            with open(f, 'r', newline='', encoding='utf8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if not row or 'original_id' in row[0].lower() or len(row) < 4:
                        continue
                    # Common CSV headers to skip
                    if any(h in row[0].lower() for h in ["original_id", "subject", "file"]):
                        continue
                    if row[3].lower() in ["subjectid", "subject", "speaker_id"]:
                        continue
                    # audio path -> row[1], subject_id -> row[3]
                    audio_path = row[1].strip()
                    subject_id = row[3].strip()
                    if audio_path and os.path.isfile(audio_path):
                        try:
                            duration, loudness = measure_audio_features(audio_path)
                            durations.append(duration)
                            loudnesses.append(loudness)
                            subject2duration[subject_id] = subject2duration.get(subject_id, 0) + duration
                        except Exception as e:
                            print(f"Error reading {audio_path}: {e}")
                    # Store subject -> audio file info
                    subject2files.setdefault(subject_id, []).append(audio_path)

        # Compute stats
        dur_min, dur_max, dur_mean, dur_std = summary_stats(durations)
        loud_min, loud_max, loud_mean, loud_std = summary_stats(loudnesses)

        print("Audio stats across train/test:")
        print(f"Total unique utts: {len(durations)}")

        print(f"Duration (s): min={dur_min:.2f}, max={dur_max:.2f}, mean={dur_mean:.2f}, std={dur_std:.2f}")
        print(f"Loudness (dBFS): min={loud_min:.2f}, max={loud_max:.2f}, mean={loud_mean:.2f}, std={loud_std:.2f}")

        # Calculate average audio count per subject
        subject_counts = [len(audios) for audios in subject2files.values() if audios]
        avg_count = sum(subject_counts) / len(subject_counts) if subject_counts else 0
        print(f"\nAverage number of audio per subject: {avg_count:.2f}")

        # Calculate total audio duration and average duration per subject
        total_duration = sum(subject2duration[subject] for subject in subject2duration if subject2duration[subject])
        avg_duration_per_subject = total_duration / len(subject2duration) if subject2duration else 0

        print(f"Total audio duration (seconds): {total_duration:.2f}")
        print(f"Average audio duration per subject (seconds): {avg_duration_per_subject:.2f}")

        # Find the 5 subjects with the fewest audio files
        # sorted_subjects = sorted(subject2files.items(), key=lambda x: len(x[1]))
        # print("\nThe 5 subjects with the fewest audio files:")
        # for subject, audios in sorted_subjects[:5]:
        #     # if subject.startswith("22"):
        #     print(f"  Subject: {subject}, Count: {len(audios)}, gender: {demo_dict.get(subject, ('', ''))[1]}, age: {demo_dict.get(subject, ('', ''))[0]}")

        # Print overall summary
        print()
        
        print("HC group:")
        print(f"  Subjects: {hc_stats['count']}")
        print(f"  Male: {hc_stats['male']}, Female: {hc_stats['female']}")
        print(f"  Average age: {hc_stats['avg_age']:.2f}")
        print(f"  Age range: {hc_stats['age_min']:.2f} - {hc_stats['age_max']:.2f}")
        print()
        
        print("PD group:")
        print(f"  Subjects: {pd_stats['count']}")
        print(f"  Male: {pd_stats['male']}, Female: {pd_stats['female']}")
        print(f"  Average age: {pd_stats['avg_age']:.2f}")
        print(f"  Age range: {pd_stats['age_min']:.2f} - {pd_stats['age_max']:.2f}")

if __name__ == "__main__":
    main()