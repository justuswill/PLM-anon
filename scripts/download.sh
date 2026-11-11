#!/bin/bash
# Download released checkpoints into checkpoints/, mirroring the repository: checkpoints/posttrain/<name>, checkpoints/pretrain/<name>.
#   bash scripts/download.sh                         # the five post-trained Table 11 models and the distilled model
#   bash scripts/download.sh posttrain/plm-1.3B-beta1
#   bash scripts/download.sh --pretrain              # also the 19 pretrained models and the two EAGLE-3 heads
set -euo pipefail

MIRROR=https://anonymous-hf.com/api/a/gk540ibws5qh   # anonymous mirror of the checkpoint repository
POSTTRAIN=(posttrain/ar-1.3B posttrain/plm-1.3B-beta0.5 posttrain/plm-1.3B-beta1 posttrain/plm-1.3B-beta2 posttrain/plm-3B-beta1 posttrain/ptp-lora-1.3B)
PRETRAIN=(pretrain/ar-350M pretrain/ar-1.3B pretrain/plm-350M-beta1 pretrain/plm-1.3B-beta0.5 pretrain/plm-1.3B-beta1 pretrain/plm-1.3B-beta2 pretrain/plm-3B-beta1
          pretrain/eagle3-plm-1.3B-beta1 pretrain/eagle3-plm-3B-beta1
          pretrain/ar-80M pretrain/plm-80M-beta0.5 pretrain/plm-80M-beta1 pretrain/plm-80M-beta2
          pretrain/ar-150M pretrain/plm-150M-beta0.5 pretrain/plm-150M-beta1 pretrain/plm-150M-beta2
          pretrain/ar-350M-23.6B pretrain/plm-350M-beta0.5-23.6B pretrain/plm-350M-beta1-23.6B pretrain/plm-350M-beta2-23.6B)
MODELS=()
EXTRA=()
for argument in "$@"; do
    case $argument in
        --pretrain) EXTRA+=("${PRETRAIN[@]}") ;;
        *) MODELS+=("$argument") ;;
    esac
done
[ ${#MODELS[@]} -gt 0 ] || MODELS=("${POSTTRAIN[@]}")

list_files() {   # "<path> <size>" for every file under one model folder
    curl -fsSL "$MIRROR/tree/$1" | python3 -c 'import json, sys; [print(f["path"], f["size"]) for f in json.load(sys.stdin) if f["type"] == "file"]'
}

download_file() {   # the mirror does not resume, so a file is fetched whole under a temporary name and kept only when complete
    local path=$1 size=$2 target=checkpoints/$1
    [ -f "$target" ] && [ "$(stat -c %s "$target")" = "$size" ] && return
    mkdir -p "$(dirname "$target")"
    curl -fL --retry 3 -o "$target.part" "$MIRROR/resolve/$path"
    mv "$target.part" "$target"
}

for model in "${MODELS[@]}" "${EXTRA[@]}"; do
    list_files "$model" | while read -r path size; do download_file "$path" "$size"; done
done
