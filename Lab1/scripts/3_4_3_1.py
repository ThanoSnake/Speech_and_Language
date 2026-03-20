#!/usr/bin/env python3
# 3.4.3.1 - Δημιουργία αρχείων uttids από τα filesets του dataset
# Εκτέλεση από: ~/kaldi/egs/usc/

import os

DATA = '/mnt/c/Users/ManosChatzigeorgiou/Documents/ntua/nlp/usc/usc'

mapping = {
    'train': 'training.txt',
    'dev':   'validation.txt',
    'test':  'testing.txt',
}

for split, filename in mapping.items():
    src = os.path.join(DATA, 'filesets', filename)
    dst = os.path.join('data', split, 'uttids')

    with open(src, 'r') as f:
        uttids = sorted([line.strip() for line in f if line.strip()])

    with open(dst, 'w') as f:
        f.write('\n'.join(uttids) + '\n')

    print(f"Created {dst} ({len(uttids)} utterances)")
