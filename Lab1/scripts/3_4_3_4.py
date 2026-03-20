#!/usr/bin/env python3
# 3.4.3.4 - Δημιουργία αρχείων text με φωνηματική μεταγραφή
# Κάθε πρόταση μετατρέπεται σε ακολουθία φωνημάτων από το lexicon.txt
# Προστίθεται sil στην αρχή και στο τέλος κάθε πρότασης
# Εκτέλεση από: ~/kaldi/egs/usc/

import os
import re

transcriptions_path = 'transcriptions.txt'
lexicon_path = 'lexicon.txt'
data_dir = 'data'
sets = ['train', 'dev', 'test']

def get_lexicon(path):
    lexicon = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) == 2:
                word = parts[0].strip().lower()
                word = re.sub(r"[^a-z' ]", '', word).strip().lower()
                phonemes = parts[1].strip()
                lexicon[word] = phonemes
    return lexicon

def get_transcriptions(path):
    trans = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) == 2:
                trans[parts[0]] = parts[1]
    return trans

def clean(text, lexicon):
    text = text.lower()
    text = text.replace('-', ' ')
    text = re.sub(r"[^a-z' ]", "", text)
    words = text.split()
    phonemes = []
    for word in words:
        if word in lexicon:
            phonemes.append(lexicon[word])
        else:
            print(f"Warning: Word '{word}' not found in lexicon.")
    return f"sil {' '.join(phonemes)} sil"

def create_text_files():
    lexicon = get_lexicon(lexicon_path)
    trans_dict = get_transcriptions(transcriptions_path)

    for s_set in sets:
        input_uttids = os.path.join(s_set, 'uttids')
        output_text = os.path.join(s_set, 'text')

        with open(input_uttids, 'r') as f_in, open(output_text, 'w') as f_out:
            for line in f_in:
                utt_id = line.strip()
                if not utt_id: continue

                sentence_id = utt_id.split('_')[-1]

                if sentence_id in trans_dict:
                    raw_text = trans_dict[sentence_id]
                    phonemized_text = clean(raw_text, lexicon)
                    f_out.write(f"{utt_id} {phonemized_text}\n")
                else:
                    print(f"Error: Sentence ID {sentence_id} does not exist in transcriptions.txt")

create_text_files()
