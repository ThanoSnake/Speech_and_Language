#!/bin/bash
# 4.4.5 - Alignment, εκπαίδευση triphone μοντέλου, HCLG και αποκωδικοποίηση
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

# Alignment με το monophone μοντέλο
steps/align_si.sh --cmd "$train_cmd" --nj 4 \
    data/train data/lang exp/mono exp/mono_ali

echo "Alignment done: exp/mono_ali/"

# Εκπαίδευση triphone μοντέλου (context-dependent phones)
# 2000 leaves (decision tree), 10000 total Gaussians
steps/train_deltas.sh --cmd "$train_cmd" \
    2000 10000 \
    data/train data/lang exp/mono_ali exp/tri1

echo "Triphone model trained: exp/tri1/"

# Δημιουργία HCLG γράφων για triphone
for lm in ug bg; do
    utils/mkgraph.sh \
        data/lang_test_${lm} \
        exp/tri1 \
        exp/tri1/graph_${lm}
    echo "HCLG graph created: exp/tri1/graph_${lm}/"
done

# Αποκωδικοποίηση dev και test
for lm in ug bg; do
    for split in dev test; do
        steps/decode.sh --cmd "$decode_cmd" --nj 4 \
            exp/tri1/graph_${lm} \
            data/$split \
            exp/tri1/decode_${split}_${lm}
        echo "Decoded $split with $lm LM -> exp/tri1/decode_${split}_${lm}/"
    done
done

# Εμφάνιση αποτελεσμάτων PER
echo ""
echo "=== Phone Error Rate (PER) - Triphone ==="
for lm in ug bg; do
    for split in dev test; do
        echo -n "  $lm / $split: "
        cat exp/tri1/decode_${split}_${lm}/scoring_kaldi/best_wer 2>/dev/null || echo "not found"
    done
done
