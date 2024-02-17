#!/usr/bin/env bash
# large files are stored in the git repository. use this script to remove them in
# git change history in order to reduce the git repository size
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/.." || exit 1

FILE="data/hist.parquet"

remove() {
  echo "remove $FILE from history"
  git filter-branch -f --index-filter "git rm -rf --cached --ignore-unmatch $FILE" HEAD
  git filter-branch -f --index-filter "git rm -rf --cached --ignore-unmatch data/info.json.gz" HEAD
  git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
}

fetch() {
  python src/market_watch/etl/__main__.py --all
}

commit() {
  git add $FILE
  git add data/info.json.gz
  git commit -m 'trimmed data and refetched'
}

push() {
  git push origin main -f
}

clean() {
  echo clean
  git reflog expire --expire=now --all
  git gc --aggressive --prune=now
}

all() {
  remove
  fetch
  commit
  push
  clean
}

while getopts ":cfhr" opt; do
  case "$opt" in
  c)
    clean
    exit 0
    ;;
  f)
    fetch
    exit 0
    ;;
  r)
    remove
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
