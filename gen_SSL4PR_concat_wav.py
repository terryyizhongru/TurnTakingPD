import os
import csv
import random
from pydub import AudioSegment

def extract_subject_id(wav_path):
    """
    Assumes the filename contains something like 'subj-1104_...' and returns the numeric part '1104'.
    Example filename: 'subj-1104_vos_2obj_mbt_men.wav_1.wav'
    """
    base_name = os.path.basename(wav_path)
    first_part = base_name.split('_')[0]  # e.g. 'subj-1104'
    return first_part.replace("subj-", "")

def concat_wavs_by_subject(example_wavs, output_folder, output_txt="merged_wavs.txt"):
    # Shuffle the list

    # Build a dictionary of subject_id -> list of wav paths
    subject_dict = {}
    for wav in example_wavs:
        sid = extract_subject_id(wav)
        subject_dict.setdefault(sid, []).append(wav)

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as f:
        # For each subject, chunk their wav files in groups of 4
        for sid, wavlist in subject_dict.items():
            for i in range(0, len(wavlist), 4):
                group = wavlist[i:i + 4]
                # Skip if not enough to form a group of 4
                if len(group) < 4:
                    break
                print("group list:", group)
                # Concatenate
                combined = AudioSegment.silent(duration=0)
                for w in group:
                    audio_part = AudioSegment.from_wav(w)
                    combined += audio_part

                # Export the merged file
                merged_filename = f"subj-{sid}_merged_{i // 4}.wav"
                merged_path = os.path.join(output_folder, merged_filename)
                combined.export(merged_path, format="wav")

                # Write the merged file path to the txt file
                f.write(merged_path + "\n")

if __name__ == "__main__":
    # Example usage
    example_wavs = []
    with open("sync_private/filelists/all_batch123_wavlists_clean.txt", "r") as f:
        example_wavs = f.read().splitlines()
    # shuffle the list
    import random
    random.seed(42)
    random.shuffle(example_wavs)
    concat_wavs_by_subject(example_wavs, output_folder="/data/storage1t/Turntaking/new_merged_wavs", output_txt="merged_wavs.txt")

    example_wavs_unnorm = [wav.replace("all_batch123", "all_batch123_unnorm") for wav in example_wavs]
    concat_wavs_by_subject(example_wavs_unnorm, output_folder="/data/storage1t/Turntaking/new_merged_wavs_unnorm", output_txt="merged_wavs_unnorm.txt")