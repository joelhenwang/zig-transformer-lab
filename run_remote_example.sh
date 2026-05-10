#!/bin/bash
# Usage: bash run_remote.sh "python3 -c 'import torch; print(torch.cuda.is_available())'"

REMOTE="joelwang-rtx@192.168.1.197"
REMOTE_DIR="~/Desktop/ai_lab/zig-transformer-lab"
VENV="~/Desktop/ai_lab/.venv"

ssh -o ConnectTimeout=10 "$REMOTE" "
  source $VENV/bin/activate
  cd $REMOTE_DIR
  $@
"
