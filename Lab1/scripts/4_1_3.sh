#!/bin/bash
# Το score.sh απαιτείται από το steps/decode.sh
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

mkdir -p local
ln -sf $KALDI_ROOT/egs/wsj/s5/steps/score_kaldi.sh local/score_kaldi.sh
ln -sf $KALDI_ROOT/egs/wsj/s5/steps/score_kaldi.sh local/score.sh

