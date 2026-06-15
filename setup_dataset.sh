#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="dataset"
ARCHIVE="$DATA_DIR/Chess Recognition Dataset (ChessReD)_2_all.zip"
URL="https://data.4tu.nl/ndownloader/items/99b5c721-280b-450b-b058-b2900b69a90f/versions/2"
EXTRACTED="$DATA_DIR/Chess Recognition Dataset (ChessReD)_2_all"

mkdir -p "$DATA_DIR"

if [ -d "$EXTRACTED/chessred/images" ] && [ -d "$EXTRACTED/chessred2k/images" ]; then
    echo "Dataset already prepared."
    exit 0
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading dataset..."
    curl -L "$URL" -o "$ARCHIVE"
fi

spinner() {
    local pid=$1
    local delay=0.25
    local frames=("[.  ]" "[ . ]" "[  .]" "[.  ]" "[.. ]" "[ ..]" "[  .]")

    stty -echo -icanon min 0 time 1
    local i=0
    while kill -0 $pid 2>/dev/null; do
        printf "\r%s  " "${frames[$i]}"
        ((i = (i + 1) % ${#frames[@]}))
        sleep $delay
    done
    printf "\b\b\b\b\b" 
    stty sane
}

echo "Extracting archive..."
set +e
unzip -o -q "$ARCHIVE" -d "$DATA_DIR" &
spinner $!
set -e

MAIN_DIR="$EXTRACTED"
mkdir -p "$MAIN_DIR"

if [ -f "$DATA_DIR/chessred.zip" ]; then
    set +e
    unzip -o -q "$DATA_DIR/chessred.zip" -d "$MAIN_DIR/chessred/" &
    spinner $!
    set -e
fi

if [ -f "$DATA_DIR/chessred2k.zip" ]; then
    set +e
    unzip -o -q "$DATA_DIR/chessred2k.zip" -d "$MAIN_DIR/chessred2k/" &
    spinner $!
    set -e
fi

# Przenieś annotations.json i inne pliki z dataset/ do MAIN_DIR/
if [ -f "$DATA_DIR/annotations.json" ]; then
    mv "$DATA_DIR/annotations.json" "$MAIN_DIR/"
fi

mv "$DATA_DIR/chessred.zip" "$MAIN_DIR/" 2>/dev/null || true
mv "$DATA_DIR/chessred2k.zip" "$MAIN_DIR/" 2>/dev/null || true

echo "Dataset ready!"