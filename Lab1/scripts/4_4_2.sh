#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

utils/mkgraph.sh data/lang_test_ug exp/mono exp/mono/graph_ug
utils/mkgraph.sh data/lang_test_bg exp/mono exp/mono/graph_bg

