#!/usr/bin/env bash
# --- manage_data.sh ---

set -e


smart_extract() {
    local src_path="$1"
    local target_path="$2"

    if [ ! -f "$src_path" ]; then
        echo "[X] Error: Source file '$src_path' not found. Skip extraction."
        exit 1
    fi

    if [ -d "$target_path" ] && [ "$(ls -A "$target_path")" ]; then
        echo "[√] $target_path already exists; skip decompression."
        return
    fi

    mkdir -p "$target_path"
    local temp_extract_dir="${target_path}_tmp_$(date +%s)"
    mkdir -p "$temp_extract_dir"
    echo "📂 Extracting from $(basename "$src_path") to $target_path ..."

    if [[ "$src_path" == *.zip ]]; then
        # -q 安静模式，-o 覆盖，-d 指定目录
        unzip -q -o "$src_path" -d "$temp_extract_dir" || { echo "Zip extraction failed"; exit 1; }
    elif [[ "$src_path" == *.rar ]]; then
        # rar 需要 unrar，-x 抽出，-o+ 覆盖，-y 自动确认
        if command -v unrar &> /dev/null; then
            unrar x -o+ -y "$src_path" "$temp_extract_dir" > /dev/null || { echo "Rar extraction failed"; exit 1; }
        else
            echo "[X] Error: unrar not found, please execute apt-get install unrar -y"
            exit 1
        fi
    fi

    local content_count=$(ls -1 "$temp_extract_dir" | wc -l)
    local sub_dir=$(ls -1 "$temp_extract_dir")

    if [ "$content_count" -eq 1 ] && [ -d "$temp_extract_dir/$sub_dir" ]; then
        echo "📦 Detected nested folder '$sub_dir', flattening..."
        mv "$temp_extract_dir/$sub_dir"/* "$target_path/" 2>/dev/null || true
    else
        mv "$temp_extract_dir"/* "$target_path/" 2>/dev/null || true
    fi

    rm -rf "$temp_extract_dir"

    # find "$target_path" -type d -exec chmod 755 {} +
    # find "$target_path" -type f -exec chmod 644 {} +
    if [ "$USER" != "" ]; then
        chown -R "$USER:$USER" "$target_path" 2>/dev/null || true
    fi
    chmod -R 755 "$target_path"

    echo -e "\033[1;34m✨ Successfully extracted and fixed permissions for $target_path\033[0m"
}


# Please configure a proxy according to your network environment before running:
proxy_on() {
    local port="${1:-7890}"
    export http_proxy="http://127.0.0.1:${port}"
    export https_proxy="${http_proxy}"
    echo "Proxy on (port ${port})"
}


proxy_off() {
    unset http_proxy https_proxy
    echo "Proxy off"
}


# Check available proxy ports
detect_proxy_port() {
    if [ -n "$http_proxy" ]; then
        local port=$(echo "$http_proxy" | sed -E 's/.*:([0-9]+).*/\1/')
        if curl -s -o /dev/null --max-time 2 --proxy "$http_proxy" "https://httpbin.org/get" 2>/dev/null; then
            echo "$port"
            return 0
        fi
    fi

    for port in 7890 10809 1080; do
        if curl -s -o /dev/null --max-time 2 --proxy "http://127.0.0.1:$port" "https://httpbin.org/get" 2>/dev/null; then
            echo "$port"
            return 0
        fi
    done

    return 1
}


# Target Dataset:  TID2013, KoNViD-1k, T2VQA-DB
# All datasets are GiB-level datasets, in imgs and videos formats, and are not ultra-large-scale.
# Therefore, training can be performed directly after decompression, without the need for streaming processing or symbolic links.
TID_SOURCE_URL="http://www.ponomarenko.info/tid2013/tid2013.rar"
KON_VIDEOS_SOURCE_URL="https://datasets.vqa.mmsp-kn.de/archives/KoNViD_1k_videos.zip"
KON_METADATA_SOURCE_URL="https://datasets.vqa.mmsp-kn.de/archives/KoNViD_1k_metadata.zip"
T2V_SOURCE_URL="https://drive.google.com/file/d/1aak5hgYsXock19d1rVufss3_X6eEA4Wx/view"


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_PARENT_DIR="$(dirname "$PROJECT_DIR")"


SEARCH_DIRS_=(
    "/root/autodl-pub/dataset"
    "/root/autodl-pub"
)
TID_DOWNLOAD_FLAG=false
KON_DATA_DOWNLOAD_FLAG=false
KON_METADATA_DOWNLOAD_FLAG=false
T2V_DOWNLOAD_FLAG=false



TID_TARGET_PATH="${PROJECT_DIR}/datasets/TID2013"
KON_DATA_TARGET_PATH="${PROJECT_DIR}/datasets/KoNViD-1k/KoNViD-1k_videos"
KON_METADATA_TARGET_PATH="${PROJECT_DIR}/datasets/KoNViD-1k/KoNViD-1k_metadata"
T2V_TARGET_PATH="${PROJECT_DIR}/datasets/T2VQA-DB"

DOWNLOAD_CACHE="${PROJECT_DIR}/.download_cache"


# Supports fuzzy matching
search_dataset() {
    local keyword="$1"
    local search_paths=("${@:2}") # Get all search root directories

    # 1. Prioritize finding the already unzipped directory.
    # -maxdepth 4: Limit search depth
    # -type d: Search only directory
    # -iname: Ignore case in name matching
    # -quit: Stop and quit as soon as it finds a match that meets the criteria.
    local found_dir=$(find "${search_paths[@]}" \
                            -maxdepth 4 \
                            -type d \
                            -iname "*${keyword}*" \
                            -print -quit 2>/dev/null)

    if [ -n "$found_dir" ]; then
        echo "FOLDER|$found_dir"
        return
    fi

    # 2. If the directory is not found, look for the compressed file.
    # -type f: Find files only
    local found_zip=$(find "${search_paths[@]}" \
                            -maxdepth 4 \
                            -type f \( -iname "*${keyword}*.zip" \
                                -o -iname "*${keyword}*.rar" \) \
                            -print -quit 2>/dev/null)

    if [ -n "$found_zip" ]; then
        echo "ARCHIVE|$found_zip"
        return
    fi

    echo "NOT_FOUND|"
}


handle_dataset_initialization() {
    local key="$1"      # Search keywords, such as "tid2013"
    local target="$2"   # Target symbolic link/extraction location
    local label="$3"    # The label name used for display

    echo "🔍 Retrieving $label ..."

    # Call your hardcore find function
    local result=$(search_dataset "$key" "${SEARCH_DIRS_[@]}")
    local status="${result%%|*}"   # Delete the first '|' character from right to left and all content to its right.
    local found_path="${result#*|}"  # Delete the first '|' character from left to right and all content to its left.

    case "$status" in
        "FOLDER")
            echo "[√] The $label directory was found: $found_path"
            ln -snf "$found_path" "$target"
            echo "    -> A symbolic link has been created to $target"
            return 0 # Success, no download required.
            ;;
        "ARCHIVE")
            echo "[√] The $label compressed file was found: $found_path"
            smart_extract "$found_path" "$target"
            echo "    -> Decompressed in place $target"
            return 0 # Success, no download required.
            ;;
        *)
            echo "[!] Not in the local area or public $label"
            return 1 # Failed, download required.
            ;;
    esac
}

handle_dataset_initialization "tid2013" "$TID_TARGET_PATH" "TID2013" || TID_DOWNLOAD_FLAG=true
handle_dataset_initialization "konvid-1k-videos" "$KON_DATA_TARGET_PATH" "KoNViD-1k" || KON_DATA_DOWNLOAD_FLAG=true
handle_dataset_initialization "konvid-1k-metadata" "$KON_METADATA_TARGET_PATH" "KoNViD-1k" || KON_METADATA_DOWNLOAD_FLAG=true
handle_dataset_initialization "t2vqa-db" "$T2V_TARGET_PATH" "T2VQA-DB" || T2V_DOWNLOAD_FLAG=true

DOWNLOAD_FLAG=false
[ "$TID_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$KON_DATA_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$KON_METADATA_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$T2V_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true



PROXY_PORT=$(detect_proxy_port)
if [ -n "$PROXY_PORT" ]; then
    proxy_on "$PROXY_PORT"      # If there is a proxy, set environment variables
else
    proxy_off                    # No agent required, ensure environment variables are empty
    echo "No proxy, using direct connection"
fi



if [ "$DOWNLOAD_FLAG" == true ]; then
    echo "[!] Some datasets are missing; download mode is now available."

    ulimit -n 65535
    echo "Initiating sequential download mode for stability..."

    if [ "$KON_DATA_DOWNLOAD_FLAG" == true ]; then
        echo "Downloading the konvid-1k videos dataset..."
        echo "Help Command: cd ./deep-vqa-framework/scripts;tail -f konvid_v.log"
        aria2c --check-certificate=false \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
            --header="Referer: https://datasets.vqa.mmsp-kn.de/" \
            --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" \
            --header="Accept-Language: en-US,en;q=0.9" \
            -x 4 -s 4 -k 2M \
            --max-tries=0 --retry-wait=10 \
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "KoNViD_1k_videos.zip" \
            "${KON_VIDEOS_SOURCE_URL}"
    fi

    if [ "$KON_METADATA_DOWNLOAD_FLAG" == true ]; then
        echo "Downloading the konvid-1k metadata dataset..."
        echo "Help Command: cd ./deep-vqa-framework/scripts;tail -f konvid_m.log"
        aria2c --check-certificate=false \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
            --header="Referer: https://datasets.vqa.mmsp-kn.de/" \
            -x 4 -s 4 -k 2M \
            --max-tries=0 --retry-wait=10 \
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "KoNViD_1k_metadata.zip" \
            "${KON_METADATA_SOURCE_URL}"
    fi

    if [ "$TID_DOWNLOAD_FLAG" == true ]; then
        echo "Downloading the tid2013 dataset..."
        echo "Help Command: cd ./deep-vqa-framework/scripts;tail -f tid.log"
        aria2c --check-certificate=false \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
            --header="Referer: http://www.ponomarenko.info/tid2013/" \
            -x 4 -s 4 -k 2M \
            --max-tries=0 --retry-wait=10 \
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "tid2013.rar" \
            "${TID_SOURCE_URL}"
    fi

    if [ "$T2V_DOWNLOAD_FLAG" == true ]; then
        echo "Downloading the t2vqa dataset..."
        echo "Help Command: cd ./deep-vqa-framework/scripts;tail -f t2vqa.log"
        if ! command -v gdown &> /dev/null; then
            uv lock --upgrade-package gdown
            uv run gdown --version
        fi
        uv run gdown -O "${DOWNLOAD_CACHE}/t2vqa.zip" \
            --continue \
            "${T2V_SOURCE_URL}"
    fi
    echo "All dataset downloads have been completed."
fi


smart_extract "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" "$KON_DATA_TARGET_PATH"
smart_extract "${DOWNLOAD_CACHE}/tid2013.rar" "$TID_TARGET_PATH"
smart_extract "${DOWNLOAD_CACHE}/t2vqa.zip" "$T2V_TARGET_PATH"
smart_extract "${DOWNLOAD_CACHE}/KoNViD_1k_metadata.zip" "$KON_METADATA_TARGET_PATH"

for dir in "$TID_TARGET_PATH" "$KON_DATA_TARGET_PATH" "$KON_METADATA_TARGET_PATH" "$T2V_TARGET_PATH"; do
    [ -d "$dir" ] && [ -n "$(ls -A "$dir")" ] || { echo "[X] $dir is empty or missing"; exit 1; }
done

echo "All datasets have been decompressed successfully."
echo "DOWNLOAD_FLAG=$DOWNLOAD_FLAG" > "./download_flag"