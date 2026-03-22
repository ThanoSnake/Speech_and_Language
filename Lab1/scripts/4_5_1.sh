#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

steps/align_si.sh --cmd "$train_cmd" --nj 4 data/train data/lang exp/tri1 exp/tri1_ali_train
steps/align_si.sh --cmd "$train_cmd" --nj 4 data/dev data/lang exp/tri1 exp/tri1_ali_dev
steps/align_si.sh --cmd "$train_cmd" --nj 4 data/test data/lang exp/tri1 exp/tri1_ali_test
