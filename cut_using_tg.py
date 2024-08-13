
import os
import glob

import textgrid
import numpy as np
from tqdm import tqdm

# Load the TextGrid file
def load_textgrid(file_path):
    try:
        tg = textgrid.TextGrid.fromFile(file_path)
        # Iterate through tiers
        for tier in tg.tiers:
            for interval in tier:
                if interval.mark != '':
                    res = (interval.minTime, interval.maxTime, interval.mark)
            

            return res, int(tier[-1].maxTime * 100)
    except Exception as e:
        print(f"An error occurred: {e} at error file: {file_path}")
        return None, None


folder = '/data/storage025/wavs_single_channel_normalized/EarlyLate/'
outdir = '/data/storage025/wavs_single_channel_normalized_nosil/EarlyLate/'
# make out folder
os.makedirs(outdir, exist_ok=True)

files = glob.glob(os.path.join(folder, '*.wav'))
diff = []
cnt_empty = 0
real_empty = 0
clean_id = open('clean_id_responsetime_EL' + '.txt', 'w')

for fn in tqdm(files):
    if 'prac' in fn:
        continue

    # wav = read_audio(fn) # backend (sox, soundfile, or ffmpeg) required!
    # if wav.shape[0] <= 0.15 * 16000:
    #     print(f"File too short: {fn}")
    #     continue
    # wav = wav[16*150:]
    # speech_timestamps = get_speech_timestamps(wav, model)

    # if len(speech_timestamps) == 0:
    #     print(f"empty: {fn}")
    #     cnt_empty += 1
    #     print(cnt_empty)
    #     continue
    # start = speech_timestamps[0]['start']/16000.0 + 0.15
    # end = speech_timestamps[-1]['end']/16000.0 + 0.15


    tgfn = os.path.basename(fn)
    tgfn = tgfn.split('.')[0] + '_' + tgfn.split('.')[1] + '.TextGrid'
    if tgfn is not None:
        tgfn = os.path.join('wavs_single_channel/EarlyLate-TG/', os.path.basename(tgfn))
    tg, length = load_textgrid(tgfn)
    if tg is None:
        continue

    if tg[2] == 'NA' or tg[1] - tg[0] < 0.25:
        cnt_empty += 1
        continue
        # print(f"Empty interval: {tgfn}")
        # print(tg)
    outf = os.path.join(outdir, os.path.basename(fn))
    os.system('sox  ' + fn + ' ' + outf + ' trim ' + str((int(tg[0] * 100)) / 100)+ ' ' + str((int( (tg[1] - tg[0] ) * 1000)) / 1000))
    clean_id.write(os.path.basename(fn) + '\t' + str(tg[0]) + '\n')


    
            
# calculate the max difference and average difference

print(f"cnt_empty: {cnt_empty}")
clean_id.close()

