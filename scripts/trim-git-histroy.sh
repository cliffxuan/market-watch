#!/usr/bin/env bash
# large files are stored in the git repository. use this script to remove them in
# git change history in order to reduce the git repository size
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/.." || exit 1

before() {
  git filter-branch -f --index-filter 'git rm -rf --cached --ignore-unmatch data/spx_hist.parquet' HEAD
  git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
}

fetch() {
  python src/market_watch/etl/__main__.py --all
}

commit() {
  git add data/spx_hist.parquet
  git add data/spx_info.json.gz
  git commit -m 'trimmed data and refetched'
}

push() {
  git push origin main -f
}

clean() {
  git reflog expire --expire=now --all
  git gc --aggressive --prune=now
}

all() {
  before
  fetch
  commit
  push
  clean
}

while getopts ":cfh" opt; do
  case "$opt" in
  c)
    clean
    exit 0
    ;;
  f)
    fetch
    exit 0
    ;;
  h)
    echo Usage "$(basename "$0") [-c | -f]"
    exit 0
    ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    exit 1
    ;;
  :)
    echo "Option -$OPTARG requires an argument." >&2
    exit 1
    ;;
  esac
done
all
