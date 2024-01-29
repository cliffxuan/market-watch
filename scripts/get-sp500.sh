#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
curl https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv > "$DIR/../data/spx_constituents.csv"
