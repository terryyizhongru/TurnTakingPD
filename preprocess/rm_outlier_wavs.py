import sys
import os

def remove_wav_paths(input_file):
    with open(input_file, 'r') as f_in:
        for line in f_in:
            if ".wav" not in line.lower():
                print("no .wav in line: " + line)
                continue
            if not os.path.exists(line.strip()) and not os.path.exists(line.strip().replace("_normalized", "")):
                print("file does not exist: " + line)
                continue
            
            if os.path.exists(line.strip()):
                os.system("rm " + line.strip())
            
            if os.path.exists(line.strip().replace("_normalized", "")):
                os.system("rm " + line.strip().replace("_normalized", ""))
                

        
                

if __name__ == "__main__":
    input_file = sys.argv[1]
    remove_wav_paths(input_file)