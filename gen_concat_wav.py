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

def concat_wavs_by_subject(example_wavs, output_folder, cat_num = 4, output_txt="merged_wavs.txt"):
    # Shuffle the list

    # Build a dictionary of subject_id -> list of wav paths
    subject_dict = {}
    for wav in example_wavs:
        sid = extract_subject_id(wav)
        subject_dict.setdefault(sid, []).append(wav)

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as f:
        # For each subject, chunk their wav files in groups of 5
        for sid, wavlist in subject_dict.items():
            for i in range(0, len(wavlist), cat_num):
                group = wavlist[i:i + cat_num]
                # Skip if not enough to form a group of 5
                if len(group) < cat_num:
                    break
                print("group list:", group)
                # Concatenate
                combined = AudioSegment.silent(duration=0)
                for w in group:
                    audio_part = AudioSegment.from_wav(w)
                    combined += audio_part

                # Export the merged file
                merged_filename = f"subj-{sid}_merged_{i // cat_num}.wav"
                merged_path = os.path.join(output_folder, merged_filename)
                combined.export(merged_path, format="wav")

                # Write the merged file path to the txt file
                f.write(merged_path + "\n")

if __name__ == "__main__":
    # Example usage
    import sys
    example_wavs = []
    with open(sys.argv[1], "r") as f:
        example_wavs = f.read().splitlines()
    # shuffle the list
    prefix = sys.argv[1].split("/")[-1].split("_")[0]
    import random
    random.seed(42)
    random.shuffle(example_wavs)
    concat_wavs_by_subject(example_wavs, output_folder="wavfolders/" + prefix + "merged_wavs", cat_num=3, output_txt="./" + prefix + "merged_wavs_list.txt")

    
    # example_wavs_unnorm = [wav.replace(foldername, foldername + "_unnorm") for wav in example_wavs]
    # concat_wavs_by_subject(example_wavs_unnorm, output_folder="wavfolders/" + prefix + "merged_wavs_unnorm", cat_num=3, output_txt="sync_private/splits/" + prefix + "merged_wavs_list_unnorm.txt")
