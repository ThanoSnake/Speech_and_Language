#!/bin/bash
# 4.2.1 - Δημιουργία αρχείων στο data/local/dict
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

DICT=data/local/dict

# Φωνήματα σιωπής
echo "sil" > $DICT/silence_phones.txt
echo "sil" > $DICT/optional_silence.txt

# Μη-σιωπηλά φωνήματα: εξαγωγή από training text, αφαίρεση sil, ταξινόμηση
awk '{for(i=2;i<=NF;i++) print $i}' data/train/text | grep -v '^sil$' | sort -u \
    > $DICT/nonsilence_phones.txt

# Λεξικό: 1-1 αντιστοίχηση φωνήματος με εαυτό (phone recognition)
echo "sil sil" > $DICT/lexicon.txt
awk '{print $1, $1}' $DICT/nonsilence_phones.txt >> $DICT/lexicon.txt

# lm_train.text: χωρίς utterance ID, με <s> και </s>
for split in train dev test; do
    awk '{printf "<s>"; for(i=2;i<=NF;i++) printf " " $i; print " </s>"}' \
        data/$split/text > data/$split/lm_train.text
done

# Κενό αρχείο extra_questions.txt
> $DICT/extra_questions.txt

echo "Dict files created in $DICT"
echo "Phoneme count: $(wc -l < $DICT/nonsilence_phones.txt) non-silence phones"
