#!/bin/bash
# 4.2.3 - Μετατροπή σε ARPA format και υπολογισμός perplexity (Ερώτημα 1)
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin

# Μετατροπή σε ARPA (φιλτράρισμα <unk> tokens)
compile-lm data/local/lm_tmp/lm_phone_ug.ilm.gz -t=yes /dev/stdout \
    | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_ug.arpa.gz

compile-lm data/local/lm_tmp/lm_phone_bg.ilm.gz -t=yes /dev/stdout \
    | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_bg.arpa.gz

# Υπολογισμός perplexity σε dev και test set (Ερώτημα 1)
echo "=== Perplexity Results ==="
for lm_suffix in ug bg; do
    echo "--- lm_phone_${lm_suffix} ---"
    for split in dev test; do
        echo -n "  $split: "
        compile-lm data/local/lm_tmp/lm_phone_${lm_suffix}.ilm.gz \
            --eval=data/$split/lm_train.text 2>&1 | grep -i "perp"
    done
done

echo "ARPA models saved in data/local/nist_lm/"
