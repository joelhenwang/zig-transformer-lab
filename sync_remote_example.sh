#!/bin/bash
# Sync uncommitted working-tree changes to the remote RTX machine.
#
# When to use:
#   - Iterating on CUDA code locally before you want to commit.
#   - Validating a fix on the remote before pushing.
#
# When NOT to use:
#   - Normal Stage 7 workflow: prefer `git push` here +
#     `bash run_remote_example.sh "git pull"` on the remote.
#     Atomic, reviewable, works across multiple dev machines.
#
# Usage:
#   bash sync_remote_example.sh
#
# `--delete` mirrors the tree: files deleted locally are also
# deleted on the remote. This keeps both sides in lockstep but
# requires that you remember the sync is authoritative. Drop
# the flag if you want additive-only behaviour.
#
# Notes on what is synced:
#   - tests/fixtures/ IS included. The PyTorch oracle's .ztlt files
#     must be on the remote for `zig build test-oracle` to work.
#   - .git/ is NOT synced. The remote has its own clone; we keep
#     its history independent so local WIP doesn't show up in
#     remote git log.
#   - .zig-cache/ and zig-out/ are NOT synced. They are build
#     artefacts; the remote rebuilds them.

set -euo pipefail

REMOTE="joelwang-rtx@192.168.1.197"
REMOTE_DIR="~/Desktop/ai_lab/zig-transformer-lab"

rsync -avz --delete \
  --exclude='.git/' \
  --exclude='.zig-cache/' \
  --exclude='zig-out/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.loss.csv' \
  --exclude='*.ckpt' \
  --exclude='build_output.txt' \
  --exclude='build_stash.txt' \
  ./ "$REMOTE:$REMOTE_DIR/"

echo "Synced working tree to $REMOTE:$REMOTE_DIR/"
