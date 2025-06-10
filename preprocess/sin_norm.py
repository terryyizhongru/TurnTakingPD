import os
import numpy as np
import soundfile as sf

def sin_norm(audio):
    x_max = np.max(np.abs(audio))
    if x_max == 0:
        return audio
    
    x_scaled = audio / x_max

    normalized_audio = np.sin(np.pi/2 * x_scaled)
    
    return normalized_audio


def main():
    input_list = 'merged_wavs_unnorm.txt'
    output_dir = 'merged_wavs_sinnormalized'
    output_list = 'merged_wavs_sinnormalized.txt'
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_list, 'r') as infile, open(output_list, 'w') as outfile:
        for line in infile:
            wav_path = line.strip()
            if not wav_path:
                continue
            if not os.path.exists(wav_path):
                print(f"File does not exist: {wav_path}")
                continue
            try:
                audio, sr = sf.read(wav_path)
                normalized_audio = sin_norm(audio)
                filename = os.path.basename(wav_path)
                normalized_path = os.path.join(output_dir, filename)
                sf.write(normalized_path, normalized_audio, sr)
                outfile.write(normalized_path + '\n')
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    main()