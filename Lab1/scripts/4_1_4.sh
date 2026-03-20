#!/bin/bash
# 4.1.4 - Δημιουργία conf/mfcc.conf
# Τα WAV αρχεία USC-TIMIT έχουν sampling rate 22050 Hz (όχι default 16000 Hz)
# Εκτέλεση από: ~/kaldi/egs/usc/

mkdir -p conf

cat > conf/mfcc.conf << 'EOF'
--use-energy=false
--sample-frequency=22050
EOF

echo "Created conf/mfcc.conf"
