#!/usr/bin/env bash
# large files are stored in the git repository. use this script to remove them in
# git change history in order to reduce the git repository size
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/.." || exit 1

FILES=(
  "data/hist.parquet"
  "data/info.json.gz"
)

precheck() {
  if ! git diff-index --quiet HEAD --; then
    echo "There are uncommitted changes. Please commit or stash them before running this script."
    exit 1
  fi
}

remove() {
  for FILE in "${FILES[@]}"; do
    echo "remove $FILE from history"
    git filter-branch -f --index-filter "git rm -rf --cached --ignore-unmatch $FILE" HEAD
  done
  git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
}

fetch() {
  ./.venv/bin/python etl/__main__.py --all
}

commit() {
  for FILE in "${FILES[@]}"; do
    git add "$FILE"
    git add "$FILE.timestamp"
  done
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
  precheck && remove && fetch && commit && push && clean
}

while getopts ":cfhpr" opt; do
  case "$opt" in
  c)
    clean
    exit 0
    ;;
  f)
    fetch
    exit 0
    ;;
  p)
    precheck
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
