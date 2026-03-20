#!/bin/bash
# 4.3.2 - Υπολογισμός στατιστικών CMVN ανά ομιλητή (Ερώτημα 2)
# Cepstral Mean and Variance Normalization: μηδενική μέση τιμή, μοναδιαία διακύμανση
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

for split in train dev test; do
    steps/compute_cmvn_stats.sh \
        data/$split exp/make_mfcc/$split mfcc
    echo "CMVN stats computed for $split"
done
