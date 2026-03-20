#!/usr/bin/env python3
# 3.4.2 - Δημιουργία φακέλων data/train, data/dev, data/test
# Εκτέλεση από: ~/kaldi/egs/usc/

import os

for split in ['train', 'dev', 'test']:
    os.makedirs(os.path.join('data', split), exist_ok=True)
    print(f"Created data/{split}/")
