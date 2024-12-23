import os
from pyannote.audio import Pipeline

def process_all_wavs(folder_path):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                diarization = pipeline(wav_path)
                rttm_path = wav_path.replace(".wav", ".rttm")
                with open(rttm_path, "w") as rttm:
                    diarization.write_rttm(rttm)
# ...existing code...

def check_rttm_files(folder_path):
    import os
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.rttm'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as file:
                    for line in file:
                        if 'SPEAKER_01' in line:
                            print(filepath)
                            break

# ...existing code...

if __name__ == "__main__":
    import sys
    folder_path = sys.argv[1]
    #process_all_wavs(folder_path)
    check_rttm_files(folder_path)