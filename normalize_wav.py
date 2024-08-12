import os
import glob

# normalize all wav files in wavs_single_channel using sox 
folder = 'wavs_single_channel/'
# find all wav files
for subf in ['BoundaryTone', 'EarlyLate', 'PictureNaming']:
    folder = os.path.join('wavs_single_channel', subf)
    wav_files = glob.glob(os.path.join(folder, '*.wav'))
    out_folder = os.path.join('wavs_single_channel_normalized', subf)
    # make out folder
    os.makedirs(out_folder, exist_ok=True)
    for fn in wav_files:
        outf = os.path.join(out_folder, fn.split('/')[-1])
        os.system('sox --norm=-0.445 ' + fn + ' ' + outf)
      