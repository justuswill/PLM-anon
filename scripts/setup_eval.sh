#!/bin/bash
# Fetch Google's IFEval instruction checkers (Apache-2.0) and the NLTK data they need. Run once.
set -euo pipefail

CHECKERS=third_party/instruction_following_eval
SOURCE=https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval
mkdir -p "$CHECKERS"
for file in instructions.py instructions_registry.py instructions_util.py; do
    curl -sSf --max-time 60 -o "$CHECKERS/$file" "$SOURCE/$file"
done
touch "$CHECKERS/__init__.py"
uv run --extra eval python -c "import nltk; [nltk.download(p, download_dir='third_party/nltk_data', quiet=True) for p in ('punkt', 'punkt_tab')]"
