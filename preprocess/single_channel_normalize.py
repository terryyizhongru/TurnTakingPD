import sys
import os


# add one args for whether normalize or not
if len(sys.argv) != 2:
    print("Usage: python single_channel.py [folder]")
    sys.exit(1)



dirn = sys.argv[1].rstrip('/') if sys.argv[1].endswith('/') else sys.argv[1] 


# normalize all wav files and change to single channel using sox to target dir wavs_norm_single_channel

output_root = dirn + '_single_channel'
if not os.path.exists(output_root):
    os.makedirs(output_root)

output_root1 = dirn + '_single_channel_normalized'
if not os.path.exists(output_root1):
    os.makedirs(output_root1)
# for subdir in os.listdir(dirn):
#     input_subdir = os.path.join(dirn, subdir)
#     output_subdir = os.path.join(output_root, subdir)

for root, dirs, files in os.walk(dirn):
    for file in files:
        if file.endswith(".wav"):
            targetdir = root.replace(dirn, output_root)
            if not os.path.exists(targetdir):
                os.makedirs(targetdir)                
            os.system("sox {} -c 1 {}".format(os.path.join(root, file), os.path.join(targetdir, file)))

            targetdir1 = root.replace(dirn, output_root1)
            if not os.path.exists(targetdir1):
                os.makedirs(targetdir1)
            os.system("sox --norm=-0.445 {} -c 1 {}".format(os.path.join(root, file), os.path.join(targetdir1, file)))
            # print("sox --norm=-0.445 {} -c 1 {}".format(os.path.join(root, file), os.path.join(targetdir1, file)))

