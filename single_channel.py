import sys
import os


# add one args for whether normalize or not
if len(sys.argv) != 3:
    print("Usage: python single_channel.py [normalize] 0 or 1 [folder]")
    sys.exit(1)

normalize = sys.argv[1]


dirn = sys.argv[2]

# normalize all wav files and change to single channel using sox to target dir wavs_norm_single_channel

for root, dirs, files in os.walk(dirn):
    for file in files:
        if file.endswith(".wav"):
            if normalize == '1':
                targetdir = root + '_single_channel_normalized'
                if not os.path.exists(targetdir):
                    os.makedirs(targetdir)                
                os.system("sox {} -c 1 {}".format(os.path.join(root, file), os.path.join(targetdir, file)))
            elif normalize == '0':
                targetdir = root + '_single_channel'
                if not os.path.exists(targetdir):
                    os.makedirs(targetdir)
                os.system("sox --norm=-0.445 {} -c 1 {}".format(os.path.join(root, file), os.path.join(targetdir, file)))

