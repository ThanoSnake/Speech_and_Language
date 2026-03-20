#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

DATA=/mnt/c/Users/ManosChatzigeorgiou/Documents/ntua/nlp/usc/usc

cp $DATA/filesets/training.txt   data/train/uttids
cp $DATA/filesets/validation.txt data/dev/uttids
cp $DATA/filesets/testing.txt    data/test/uttids
