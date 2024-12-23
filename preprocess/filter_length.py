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