
1. single channel and normalize, with normalization and without
    python preprocess/single_channel_normalize.py [folder]
    output: generate two folder, one with normalization one without

2. cut 2 folder using vad
    python preprocess/cut_using_vad.py [folder] [folder_normalized]
    output: generate two folder with nosil

3. check_and_filter_length of outlier
    python preprocess/check_and_filter_length.py [folder]
    output: generate filelist with error length

4. (optional) spk_diarization.py
    python preprocess/spk_diarization.py [folder]
    output: generate filelist with non-speaker and multispeaker

5. remove error file with list
    python rm_outlier_wavs.py [folder]