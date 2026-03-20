#!/bin/bash
# Δημιουργία conf/mfcc.conf
# Εκτέλεση από: ~/kaldi/egs/usc/

mkdir -p conf

cat > conf/mfcc.conf << 'EOF'
--use-energy=false
--sample-frequency=22050
EOF

