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
    

# Example usage:
folder = sys.argv[1]
check_wav_lengths(folder)