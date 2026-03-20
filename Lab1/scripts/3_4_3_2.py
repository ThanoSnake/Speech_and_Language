#!/usr/bin/env python3
# Εκτέλεση από: ~/kaldi/egs/usc/

import os

folders = ['train', 'test', 'dev']

def create_utt2spk():
    for folder in folders:
        uttids_path = os.path.join('data', folder, 'uttids')
        utt2spk_path = os.path.join('data', folder, 'utt2spk')

        if not os.path.exists(uttids_path):
            continue

        with open(uttids_path, 'r') as f_in, open(utt2spk_path, 'w') as f_out:
            for line in f_in:
                utt_id = line.strip()
                if utt_id:
                    speaker_id = utt_id[:2]
                    f_out.write(f"{utt_id} {speaker_id}\n")

create_utt2spk()
