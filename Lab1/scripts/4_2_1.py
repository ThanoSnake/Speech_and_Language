#!/usr/bin/env python3
# Εκτέλεση από: ~/kaldi/egs/usc/

DATA = '/mnt/c/Users/ManosChatzigeorgiou/Documents/ntua/nlp/usc/usc'
DICT = 'data/local/dict'

with open(f'{DICT}/silence_phones.txt', 'w') as f:
    f.write('sil\n')

with open(f'{DICT}/optional_silence.txt', 'w') as f:
    f.write('sil\n')

phones = set()
with open(f'{DATA}/lexicon.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            for p in parts[1:]:
                if p.lower() != 'sil' and p != '<oov>':
                    phones.add(p)

sorted_phones = sorted(phones)
with open(f'{DICT}/nonsilence_phones.txt', 'w', encoding='utf-8') as f:
    for p in sorted_phones:
        f.write(p + '\n')

with open(f'{DICT}/lexicon.txt', 'w', encoding='utf-8') as f:
    f.write('sil sil\n')
    for p in sorted_phones:
        f.write(f'{p} {p}\n')

for split in ['train', 'dev', 'test']:
    with open(f'data/{split}/text', 'r') as f_in, open(f'{DICT}/lm_{split}.text', 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            phones_seq = ' '.join(parts[1:])  # αφαίρεση utterance ID
            f_out.write(f'<s> {phones_seq} </s>\n')

open(f'{DICT}/extra_questions.txt', 'w').close()

print(f'Non-silence phones: {len(sorted_phones)}')
