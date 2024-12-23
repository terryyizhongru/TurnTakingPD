import os
import sys
import wave
import contextlib
import numpy as np
import os
import shutil
from pathlib import Path

def check_wav_lengths(folder_path, copy=False, short_thresh=1.0, long_thresh=10.0):
    durations = []
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]

    for wav_file in wav_files:
        filepath = os.path.join(folder_path, wav_file)
        with contextlib.closing(wave.open(filepath, 'r')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            durations.append((wav_file, duration))

    # Extract durations separately for analysis
    length_values = [d[1] for d in durations]
    min_length = min(length_values)
    max_length = max(length_values)
    mean_length = np.mean(length_values)
    std_length = np.std(length_values)

    print(f"Min length: {min_length:.2f}s")
    print(f"Max length: {max_length:.2f}s")
    print(f"Mean length: {mean_length:.2f}s")
    print(f"median length: {np.median(length_values):.2f}s")
    print(f"Std Dev: {std_length:.2f}s")

    long_thresh = mean_length + 3 * std_length
    # Identify outliers by thresholds
    
    # get top 5 short wav files
    # Sort by duration and take the first 5 entries
    top_5_short = sorted(durations, key=lambda x: x[1])[:50]
    for f, d in top_5_short:
        print(f"Short file: {f}, duration: {d:.2f}s")


    long_outliers = [(f, d) for f, d in durations if d > long_thresh]

    # Alternatively, you can also do outlier detection based on standard deviations
    # short_outliers = [f for f, d in durations if d < (mean_length - 2 * std_length)]
    # long_outliers = [f for f, d in durations if d > (mean_length + 2 * std_length)]

    print(f"\nLong outliers (over {long_thresh}s):")
    for f, d in long_outliers:
        print(f"  {f}, duration: {d:.2f}s")
    
    if copy:
        output_folder = "./check" + "_".join(folder.strip("/").split("/")[-2:])  # Change as needed
        print(f"\nCopying files to {output_folder}")
        short_outdir = Path(output_folder) / "short"
        long_outdir = Path(output_folder) / "long"
        
        # Create the subfolders if they don't exist
        short_outdir.mkdir(parents=True, exist_ok=True)
        long_outdir.mkdir(parents=True, exist_ok=True)
        
        # Copy short files
        for f, d in top_5_short:
            src = Path(folder_path) / f
            dst = short_outdir / f
            shutil.copy2(src, dst)
        
        # Copy long outliers
        for f, d in long_outliers:
            src = Path(folder_path) / f
            dst = long_outdir / f
            shutil.copy2(src, dst)
    
    

import os
import wave

def filter_audio_files(base_path):
    subdirs = {
        "PictureNaming": (2.15, None),   # > 2.15
        "EarlyLate": (3.0, None),       # > 3.0
        "BoundaryTone": (0.6, 20.0)     # < 0.6 or > 20
    }
    suspicious = []
    for subdir, (min_thresh, max_thresh) in subdirs.items():
        full_path = os.path.join(base_path, subdir)
        for root, _, files in os.walk(full_path):
            for f in files:
                if f.lower().endswith(".wav"):
                    file_path = os.path.join(root, f)
                    with wave.open(file_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        duration = frames / float(rate)
                        if subdir == "BoundaryTone":
                            if duration < min_thresh or duration > max_thresh:
                                suspicious.append((file_path, duration))
                        else:
                            if duration > min_thresh:
                                suspicious.append((file_path, duration))
    return suspicious

if __name__ == "__main__":
    
    # # Example usage of check length:
    # folder = sys.argv[1]
    # check_wav_lengths(folder)
    
    # Example usage of filter_audio_files:
    
    import sys
    base_path = sys.argv[1]
    suspicious = filter_audio_files(base_path)
    output_path = base_path.strip('/').split('/')[-1] + "outlier.txt"
    print(f"Writing to {output_path}")
    with open(output_path, "w") as f:
        for s in suspicious:
            f.write(f"{s[0]}\n")
    # for s in suspicious:
    #     print(s)
