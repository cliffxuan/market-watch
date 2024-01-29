#!/usr/bin/env bash
# large files are stored in the git repository. use this script to remove them in
# git change history in order to reduce the git repository size
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/.." || exit 1

git filter-branch -f --index-filter 'git rm -rf --cached --ignore-unmatch data/spx_hist.parquet' HEAD
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --aggressive --prune=now
