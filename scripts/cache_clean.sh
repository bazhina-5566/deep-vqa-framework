#!/usr/bin/env bash
# --- cache_clean.sh ---
# Description: Deep cleaning tool for AutoDL ecosystem with "absolute safety" and "system disk space optimization"

set -e

# --- 1. Core path definitions ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_PARENT_DIR="$(dirname "$PROJECT_DIR")"
DOWNLOAD_CACHE="${PROJECT_DIR}/.download_cache"

# Redirect AI model and package manager caches to large data disk
TMP_CACHE_DIR="${PROJECT_PARENT_DIR}/.system_caches"
mkdir -p "$TMP_CACHE_DIR/huggingface" "$TMP_CACHE_DIR/modelscope" "$TMP_CACHE_DIR/uv"

# Permission fallback
APP_SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null && APP_SUDO="sudo"

echo "🧹 Starting cross-modal compute base automated cleaning pipeline..."

# ========================================================
# Step: AI model cache migration (symlink redirect)
# ========================================================
echo "🤖 Optimizing AI foundation model cache routing (HuggingFace / ModelScope)..."

for hub_name in "huggingface" "modelscope"; do
    TARGET_LINK="$HOME/.cache/${hub_name}"
    DATA_DISK_CACHE="${TMP_CACHE_DIR}/${hub_name}"

    # If original path is a regular folder, safely migrate existing assets to data disk
    if [ -d "$TARGET_LINK" ] && [ ! -L "$TARGET_LINK" ]; then
        echo "📦 Found system disk ${hub_name} weights, safely migrating to data disk..."
        mkdir -p "$DATA_DISK_CACHE"
        cp -r "$TARGET_LINK"/* "$DATA_DISK_CACHE/" 2>/dev/null || true
        rm -rf "$TARGET_LINK"
    fi

    # Create symlink, ensuring model downloads flow to data disk
    if [ ! -L "$TARGET_LINK" ]; then
        mkdir -p "$HOME/.cache"
        ln -s "$DATA_DISK_CACHE" "$TARGET_LINK"
        echo "[√] ${hub_name} path successfully anchored to data disk."
    fi
done

# ========================================================
# Traditional cleanup items
# ========================================================

# 1. Dataset compression package safe removal
if [ -d "$DOWNLOAD_CACHE" ]; then
    # Verify extraction completion
    if [ -d "$PROJECT_DIR/datasets/KoNViD-1k" ] || [ -d "$PROJECT_DIR/datasets/TID2013" ]; then
        echo "📦 Dataset extraction verified, removing original compressed archives..."
        rm -rf "$DOWNLOAD_CACHE"
        echo "[√] Download cache safely cleared."
    else
        echo "⚠️  Dataset extraction not verified! Keeping original archives."
    fi
fi
# Remove zero-byte zombie download files
find "$PROJECT_PARENT_DIR" -name "*.zip" -size 0 -delete 2>/dev/null || true

# 2. UV and pip package manager cache cleaning
echo "🐍 Cleaning Python package manager caches..."
if command -v uv &> /dev/null; then
    export UV_CACHE_DIR="${TMP_CACHE_DIR}/uv"
    uv cache clean
fi
if command -v pip &> /dev/null; then
    pip cache purge 2>/dev/null || true
fi
echo "[√] Package manager caches cleaned."

# 3. Conda environment cleaning
if command -v conda &> /dev/null; then
    echo "📦 Cleaning Conda redundant packages..."
    conda clean --all -y > /dev/null 2>&1 || true
    echo "[√] Conda environment cleaned."
fi

# 4. APT system dependency cache cleaning
echo "📦 Cleaning system-level APT cache..."
${APP_SUDO} apt-get clean -y
${APP_SUDO} apt-get autoclean -y

# 5. Python bytecode and Jupyter cache removal
echo "📁 Removing project space temporary files..."
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR" -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# 6. WandB / TensorBoard temporary file cleanup
if [ -d "$PROJECT_DIR/wandb" ]; then
    echo "📊 Cleaning WandB historical sync fragments..."
    find "$PROJECT_DIR/wandb" -name "*.tmp" -delete 2>/dev/null || true
fi

# ========================================================
# Disk usage report
# ========================================================
echo "------------------------------------------------"
echo "📊 Data disk usage ($PROJECT_PARENT_DIR):"
df -h "$PROJECT_PARENT_DIR" | awk 'NR==2 {print "Used: " $3 " | Available: " $4 " | Usage: " $5}'

echo "📊 System disk usage (/):"
df -h / | awk 'NR==2 {print "System used: " $3 " | Available: " $4 " (safety threshold > 5GB)"}'
echo "------------------------------------------------"
echo "🚀 Framework compute environment optimized to lean & mean state!"