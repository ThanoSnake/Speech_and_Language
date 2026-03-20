#!/bin/bash
# 4.4.3 - Αποκωδικοποίηση με monophone μοντέλο (αλγόριθμος Viterbi)
# Αποκωδικοποίηση dev και test για unigram και bigram LM
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

for lm in ug bg; do
    for split in dev test; do
        steps/decode.sh --cmd "$decode_cmd" --nj 4 \
            exp/mono/graph_${lm} \
            data/$split \
            exp/mono/decode_${split}_${lm}
        echo "Decoded $split with $lm LM -> exp/mono/decode_${split}_${lm}/"
    done
done

# Εμφάνιση αποτελεσμάτων PER
echo ""
echo "=== Phone Error Rate (PER) - Monophone ==="
for lm in ug bg; do
    for split in dev test; do
        echo -n "  $lm / $split: "
        cat exp/mono/decode_${split}_${lm}/scoring_kaldi/best_wer 2>/dev/null || echo "not found"
    done
done
