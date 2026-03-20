#!/bin/bash
# 4.1.1 - Δημιουργία path.sh και cmd.sh
# Εκτέλεση από: ~/kaldi/egs/usc/

cat > path.sh << 'EOF'
export KALDI_ROOT=/home/feida/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "common_path.sh missing" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONUNBUFFERED=1
EOF

cat > cmd.sh << 'EOF'
export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl
EOF

echo "Created path.sh and cmd.sh"
