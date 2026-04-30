#!/bin/bash

# USAGE: ./run_quant.sh [strategy_number] [use_csv_boolean]
# Example: ./run_quant.sh 2 true

# Help message
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [strategy] [csv_toggle]"
    echo "  strategy:   Integer (default: 1)"
    echo "  csv_toggle: true/false (default: false)"
    exit 0
fi

# Assign arguments with defaults
STRATEGY=${1:-1}
USE_CSV=${2:-false}

# Set flag only if true
CSV_FLAG=""
[[ "$USE_CSV" == "true" ]] && CSV_FLAG="--csv"

if [ "$USE_CSV" == 'false' ]; then
	echo "--> Quants with Strategy: $STRATEGY | CSV: $USE_CSV"
fi

# Iterate through both k and v buffers
for f in data/fp16/*[kv].bin; do
    if [ -f "$f" ]; then
        ./quant --input "$f" --strategy "$STRATEGY" $CSV_FLAG
    fi
done
