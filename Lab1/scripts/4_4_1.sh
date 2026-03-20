#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

steps/train_mono.sh --cmd "$train_cmd" --nj 4 data/train data/lang exp/mono

