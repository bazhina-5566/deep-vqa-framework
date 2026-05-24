#!/usr/bin/env bash
# CI helper: test smart_extract() from manage_data.sh without downloading datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load only smart_extract (not the rest of manage_data.sh which may download data)
eval "$(sed -n '/^smart_extract() {/,/^}$/p' "${SCRIPT_DIR}/manage_data.sh")"

if ! command -v zip >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; then
  echo "FAIL: zip and unzip are required"
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# Zip layout: single top-level folder "nested_dir/" → smart_extract flattens to target root
mkdir -p "${WORKDIR}/nested_dir"
echo "mock" > "${WORKDIR}/nested_dir/img001.png"
echo "mock" > "${WORKDIR}/nested_dir/mos.csv"
( cd "${WORKDIR}" && zip -qr mock_dataset.zip nested_dir )

TARGET="${WORKDIR}/extract_output"
smart_extract "${WORKDIR}/mock_dataset.zip" "${TARGET}"

[ -f "${TARGET}/img001.png" ] || { echo "FAIL: flatten - img001.png missing"; ls -laR "${TARGET}"; exit 1; }
[ -f "${TARGET}/mos.csv" ]    || { echo "FAIL: flatten - mos.csv missing"; exit 1; }
echo "PASS: smart_extract flatten logic"

TARGET2="${WORKDIR}/already_exists"
mkdir -p "${TARGET2}"
echo "existing" > "${TARGET2}/existing.txt"
smart_extract "${WORKDIR}/mock_dataset.zip" "${TARGET2}"
[ -f "${TARGET2}/existing.txt" ] || { echo "FAIL: skip - existing file overwritten"; exit 1; }
echo "PASS: smart_extract skips non-empty existing directory"
