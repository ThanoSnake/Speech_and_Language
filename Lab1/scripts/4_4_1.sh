#!/bin/bash
# 4.4.1 - Εκπαίδευση monophone GMM-HMM ακουστικού μοντέλου
# Χρησιμοποιεί flat start και Baum-Welch EM για εκπαίδευση
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

steps/train_mono.sh --cmd "$train_cmd" --nj 4 \
    data/train data/lang exp/mono

echo "Monophone model trained in exp/mono/"
