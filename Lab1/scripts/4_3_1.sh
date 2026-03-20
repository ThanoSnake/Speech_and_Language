#!/bin/bash
# 4.3.1 - Εξαγωγή MFCC χαρακτηριστικών και για τα 3 sets
# Χρησιμοποιεί τις παραμέτρους από conf/mfcc.conf (22050 Hz, no energy)
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

for split in train dev test; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 \
        data/$split exp/make_mfcc/$split mfcc
    echo "MFCCs extracted for $split"
done
