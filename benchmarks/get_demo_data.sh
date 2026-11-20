#!/bin/bash
# Downloads the public demo breast dataset (Zenodo record 19086853, 65 MB) used by demo B and by the `real` benchmark case.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$HERE/data"
OUT="$HERE/data/demo_breast_data.h5"
[ -s "$OUT" ] && { echo "already present: $OUT"; exit 0; }
curl -L --fail --progress-bar -o "$OUT" "https://zenodo.org/records/19086853/files/demo_breast_data.h5?download=1"
ls -la "$OUT"
