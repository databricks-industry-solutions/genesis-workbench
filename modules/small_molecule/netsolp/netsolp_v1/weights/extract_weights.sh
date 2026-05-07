#!/usr/bin/env bash
# Extract the two NetSolP-1.0 distilled-solubility files we need from the
# upstream 5.63 GB tarball, into the directory this script lives in.
#
# Usage:
#   bash extract_weights.sh /path/to/netsolp-1.0.ALL.tar.gz
#
# After the script runs, commit the two extracted files:
#   git add weights/S_Distilled_quantized.onnx weights/ESM12_alphabet.pkl

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/netsolp-1.0.ALL.tar.gz"
    exit 1
fi

TARBALL="$1"
if [[ ! -f "$TARBALL" ]]; then
    echo "🚫 Tarball not found: $TARBALL"
    exit 1
fi

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Extracting NetSolP distilled weights into $DEST_DIR"

# The tarball lays out files under different prefixes depending on the release;
# match by basename so we don't have to hard-code the directory layout.
TARGETS=("Solubility_ESM12_0_quantized.onnx" "ESM12_alphabet.pkl")

for fname in "${TARGETS[@]}"; do
    echo "▶️  Extracting $fname"
    # --wildcards matches by glob; --strip-components=99 flattens path
    tar -xzf "$TARBALL" \
        --wildcards "*/${fname}" \
        --strip-components=99 \
        -C "$DEST_DIR" \
        --transform="s|.*/|${DEST_DIR}/|" 2>/dev/null \
    || tar -xzf "$TARBALL" --wildcards "*${fname}" -O > "$DEST_DIR/$fname"

    if [[ ! -s "$DEST_DIR/$fname" ]]; then
        echo "🚫 Failed to extract $fname from $TARBALL"
        echo "    Inspect the tarball manually:  tar -tzf $TARBALL | grep $fname"
        exit 1
    fi
    size_mb=$(du -m "$DEST_DIR/$fname" | cut -f1)
    echo "   ✅ $fname  (${size_mb} MB)"
done

echo
echo "Done. Now commit the two files:"
echo "  git add modules/small_molecule/netsolp/netsolp_v1/weights/Solubility_ESM12_0_quantized.onnx \\"
echo "          modules/small_molecule/netsolp/netsolp_v1/weights/ESM12_alphabet.pkl"
