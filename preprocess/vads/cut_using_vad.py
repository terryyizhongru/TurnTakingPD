import os
import glob
import sys

import textgrid
import numpy as np
from tqdm import tqdm

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


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


model = load_silero_vad()

if len(sys.argv) != 2:
    print("Usage: python cut_using_vad.py [folder]")
    sys.exit(1)

folder = sys.argv[1][:-1] if sys.argv[1].endswith('/') else sys.argv[1]
outdir = folder[:-1]+ '_nosil' if folder.endswith('/') else folder + '_nosil'   

for subdir in ['BoundaryTone', 'PictureNaming', 'EarlyLate']:
# for subdir in ['BoundaryTone']:
    wavfolder = os.path.join(folder, subdir) if 'normalized' in folder else os.path.join(folder + '_normalized', subdir) 
    os.makedirs(wavfolder, exist_ok=True)

    

    files = glob.glob(os.path.join(wavfolder, '*.wav'))
    diff = []
    cnt_empty = 0
    real_empty = 0
    clean_id = open('clean_id_responsetime_' + subdir + '.txt', 'w')

    for fn in tqdm(files):
        if 'prac' in fn:
            continue
        
        wav = read_audio(fn) # backend (sox, soundfile, or ffmpeg) required!
        if wav.shape[0] <= 0.15 * 16000:
            print(f"File too short: {fn}")
            cnt_empty += 1
            print(cnt_empty)

            continue
        
        wav = wav[16*150:]
        speech_timestamps = get_speech_timestamps(wav, model)

        if len(speech_timestamps) == 0:
            print(f"empty: {fn}")
            cnt_empty += 1
            print(cnt_empty)
            continue
        
        start = speech_timestamps[0]['start']/16000.0 + 0.15
        end = speech_timestamps[-1]['end']/16000.0 + 0.15
                
        if end - start < 0.25:
            cnt_empty += 1
            print(f"empty < 0.25: {fn}")
            print(cnt_empty)
            continue

        wavoutfolder = os.path.join(outdir, subdir)
        os.makedirs(wavoutfolder, exist_ok=True)
        clean_id.write(os.path.basename(fn) + '\t' + str(start) + '\n')
        continue
        if 'normalized' in outdir:
            outf = os.path.join(wavoutfolder, os.path.basename(fn))
            os.system('sox  ' + fn + ' ' + outf + ' trim ' + str((int(start * 100)) / 100)+ ' ' + str((int( (end - start ) * 1000)) / 1000))
            clean_id.write(os.path.basename(fn) + '\t' + str(start) + '\n')

        if 'normalized' not in outdir:
            outdir2 = outdir.replace('_nosil', '_normalized_nosil')
            wavoutfolder2 = os.path.join(outdir2, subdir)
            os.makedirs(wavoutfolder2, exist_ok=True)
            outf2 = os.path.join(wavoutfolder2, os.path.basename(fn))
            os.system('sox  ' + fn + ' ' + outf2 + ' trim ' + str((int(start * 100)) / 100)+ ' ' + str((int( (end - start ) * 1000)) / 1000))
    

            fn = fn.replace('_normalized', '')
            outf = os.path.join(wavoutfolder, os.path.basename(fn))

            os.system('sox  ' + fn + ' ' + outf + ' trim ' + str((int(start * 100)) / 100)+ ' ' + str((int( (end - start ) * 1000)) / 1000))
            clean_id.write(os.path.basename(fn) + '\t' + str(start) + '\n')



    
            
# calculate the max difference and average difference

    print(f"cnt_empty: {cnt_empty}")
    clean_id.close()


