#!/bin/bash
# USAGE: ./run.sh [corpus] [strategy] [csv_bool]
# corpus: wikitext | c4 (default: wikitext)
# Example: ./run.sh c4 2 true

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
	echo "Usage: $0 [corpus] [strategy] [csv_toggle]"
	echo "  corpus:     wikitext|c4 (default: wikitext)"
	echo "  strategy:   integer (default: 1)"
	echo "  csv_toggle: true|false (default: false)"
	exit 0
fi

CORPUS=${1:-wikitext}
STRATEGY=${2:-1}
USE_CSV=${3:-false}

DATA_DIR="data/${CORPUS}_fp16"
CSV_OUT="quant_${CORPUS}_s${STRATEGY}.csv"

if [ ! -d "$DATA_DIR" ]; then
	echo "error: $DATA_DIR not found" >&2
	exit 1
fi

CSV_FLAG=""
[[ "$USE_CSV" == "true" ]] && CSV_FLAG="--csv"

echo "--> corpus=$CORPUS strategy=$STRATEGY csv=$USE_CSV"

if [[ "$USE_CSV" == "true" ]]; then
	for f in "$DATA_DIR"/*[kv].bin; do
		[ -f "$f" ] && ./quant --input "$f" --strategy "$STRATEGY" $CSV_FLAG
	done > "$CSV_OUT"
	echo "--> wrote $CSV_OUT"
else
	for f in "$DATA_DIR"/*[kv].bin; do
		[ -f "$f" ] && ./quant --input "$f" --strategy "$STRATEGY"
	done
fi
