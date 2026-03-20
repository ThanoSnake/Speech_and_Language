#!/bin/bash
# 4.1.2 - Δημιουργία symbolic links steps και utils που δείχνουν στη διαδικασία wsj
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

ln -sf $KALDI_ROOT/egs/wsj/s5/steps steps
ln -sf $KALDI_ROOT/egs/wsj/s5/utils utils

echo "Created symlinks: steps -> wsj/s5/steps, utils -> wsj/s5/utils"
