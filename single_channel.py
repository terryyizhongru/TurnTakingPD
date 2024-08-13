import sys
import os




dirn = "/scratch/yzhong/Turntaking/"	

# normalize all wav files and change to single channel using sox to target dir wavs_norm_single_channel

for root, dirs, files in os.walk(dirn):
    for file in files:
        if file.endswith(".wav"):
            targetdir = os.path.join(root.replace('Turntaking', 'Turntaking/wavs_norm_single_channel'))
            # generate the target dir
            if not os.path.exists(targetdir):
                os.makedirs(targetdir)
            os.system("sox {} -c 1 {}".format(os.path.join(root, file), os.path.join(targetdir, file)))
            

