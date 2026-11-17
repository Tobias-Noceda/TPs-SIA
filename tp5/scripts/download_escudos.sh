#!/usr/bin/env bash
# Downloads the 64x64 logos of every club listed at https://football-logos.cc/argentina/ into the
# escudos dataset of ex. 2, named like the files already there: argentina_<id>_64x64.football-logos.cc.png
#
# The league page only lists the clubs; the hash of each 64x64 PNG is in the club's own page
# (<option value="64::<hash>">), and the image is at assets.football-logos.cc/logos/argentina/64x64/<id>.<hash>.png.
# Files that already exist are skipped, so it can be re-run safely.
#
# Usage (from tp5/): scripts/download_escudos.sh [output dir, default res/datasets/escudos]
#   LIMIT=n  only the first n clubs (for a test)
#   DELAY=s  seconds between clubs (default 0.5, to be gentle with the site)
set -euo pipefail

OUTPUT="${1:-res/datasets/escudos}"
LIMIT="${LIMIT:-0}"
DELAY="${DELAY:-0.5}"
SITE="https://football-logos.cc"
ASSETS="https://assets.football-logos.cc/logos/argentina/64x64"
AGENT="Mozilla/5.0"

mkdir -p "$OUTPUT"

ids=$(wget -q -U "$AGENT" -O - "$SITE/argentina/" | grep -oE 'data-logo-id="[^"]+"' | sed -E 's/data-logo-id="([^"]+)"/\1/' | sort -u)
if [[ -z "$ids" ]]; then
    echo "No logos found in $SITE/argentina/ (did the page change?)" >&2
    exit 1
fi
if (( LIMIT > 0 )); then
    ids=$(echo "$ids" | head -n "$LIMIT")
fi
total=$(echo "$ids" | wc -l)
echo "$total logos listed; saving into $OUTPUT"

downloaded=0 skipped=0 failed=0 index=0
for id in $ids; do
    index=$((index + 1))
    target="$OUTPUT/argentina_${id}_64x64.football-logos.cc.png"
    if [[ -f "$target" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    hash=$(wget -q -U "$AGENT" -O - "$SITE/argentina/$id/" | grep -oE 'value="64::[0-9a-f]+"' | head -n 1 | sed -E 's/.*::([0-9a-f]+)"/\1/' || true)
    if [[ -z "$hash" ]] || ! wget -q -U "$AGENT" -O "$target.part" "$ASSETS/$id.$hash.png"; then
        echo "[$index/$total] FAILED $id" >&2
        rm -f "$target.part"
        failed=$((failed + 1))
    else
        mv "$target.part" "$target"
        echo "[$index/$total] $id"
        downloaded=$((downloaded + 1))
    fi
    sleep "$DELAY"
done

echo "Done: $downloaded downloaded, $skipped already there, $failed failed ($(find "$OUTPUT" -name '*.png' | wc -l) PNG in $OUTPUT)"
