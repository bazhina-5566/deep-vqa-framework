#!/usr/bin/env bash
# --- system_check.sh ---

# Tasks:
# 1. Process check
# 2. Port check
# 3. Memory check
# 4. Video memory check
# 5. Disk space usage check
# 6. pip, pyenv, virtualenv, venv check
# 7. conda, micromamba check
# 8. UV check
# 9. http, https, apt mirror, etc. check
# 10. Permission check
# 11. Python interpreter check

set -e

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}=== Deep-VQA System Health Check ===${NC}"
echo -e "${BLUE}========================================${NC}"

# 1. Process check (training process)
echo -e "\n${YELLOW}[1/11] Checking Python training processes...${NC}"
TRAIN_PROCESS=$(ps -ef | grep -E "main.py|train.py" | grep -v "grep" | grep -v "system_check" || true)
if [ -z "$TRAIN_PROCESS" ]; then
    echo -e "${RED}[X] No running training process found!${NC}"
else
    echo -e "${GREEN}[√] Running training processes:${NC}"
    echo "$TRAIN_PROCESS" | awk '{print "  PID: " $2 " | CMD: " $8 " " $9 " " $10}'
fi
ulimit -n

# 2. Port check (common training ports)
echo -e "\n${YELLOW}[2/11] Checking common ports...${NC}"
COMMON_PORTS=(6006 8888 8080 8000 22 7860)
for port in "${COMMON_PORTS[@]}"; do
    PORT_CHECK=$(ss -tlnp 2>/dev/null | grep -q ":$port " && echo "LISTENING" || echo "FREE")
    if [ "$PORT_CHECK" = "LISTENING" ]; then
        echo -e "  Port ${port}: ${GREEN}LISTENING${NC}"
    else
        echo -e "  Port ${port}: ${YELLOW}FREE${NC}"
    fi
done

# 3. Memory check
echo -e "\n${YELLOW}[3/11] Memory usage status:${NC}"
MEM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
MEM_USED=$(free -h | awk '/^Mem:/ {print $3}')
MEM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
MEM_PERCENT=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
if (( $(echo "$MEM_PERCENT > 90" | bc -l) )); then
    MEM_COLOR=$RED
elif (( $(echo "$MEM_PERCENT > 70" | bc -l) )); then
    MEM_COLOR=$YELLOW
else
    MEM_COLOR=$GREEN
fi
echo -e "  Total: ${MEM_TOTAL} | Used: ${MEM_USED} | Available: ${MEM_AVAIL} | Usage: ${MEM_COLOR}${MEM_PERCENT}%${NC}"

# 4. GPU memory check (AutoDL essential)
echo -e "\n${YELLOW}[4/11] GPU memory status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    if [ -n "$(nvidia-smi -L 2>/dev/null)" ]; then
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=', ' read -r gpu_id gpu_util mem_used mem_total gpu_temp; do
            MEM_PERCENT=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc)
            if (( $(echo "$MEM_PERCENT > 90" | bc -l) )); then
                MEM_COLOR=$RED
            elif (( $(echo "$MEM_PERCENT > 70" | bc -l) )); then
                MEM_COLOR=$YELLOW
            else
                MEM_COLOR=$GREEN
            fi
            echo -e "  GPU $gpu_id: Util ${gpu_util}% | Memory ${mem_used}MiB / ${mem_total}MiB (${MEM_COLOR}${MEM_PERCENT}%${NC}) | Temp ${gpu_temp}°C"
        done
    else
        echo -e "  ${YELLOW}[!] No GPU detected or nvidia-smi failed.${NC}"
    fi
else
    echo -e "  ${YELLOW}[!] nvidia-smi not found (no GPU or driver not installed)${NC}"
fi

# 5. Disk space check
echo -e "\n${YELLOW}[5/11] Disk space usage:${NC}"
df -h / /root/autodl-tmp 2>/dev/null | awk 'NR==1 {print "  Filesystem      Size  Used  Avail  Use% Mount"} NR>1 {printf "  %-15s %-5s %-5s %-5s %-4s %s\n", $1, $2, $3, $4, $5, $6}'

# 6. pip, pyenv, virtualenv, venv check
echo -e "\n${YELLOW}[6/11] Python environment managers:${NC}"
for tool in pip pyenv virtualenv; do
    if command -v $tool &> /dev/null; then
        VERSION=$($tool --version 2>/dev/null | head -1)
        echo -e "  ${GREEN}[√] $tool: $VERSION${NC}"
    else
        echo -e "  ${YELLOW}[!] $tool not found${NC}"
    fi
done
# Check active venv
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "  ${GREEN}[√] Active venv: $VIRTUAL_ENV${NC}"
else
    echo -e "  ${YELLOW}[!] No active virtual environment detected${NC}"
fi

# 7. conda, micromamba check
echo -e "\n${YELLOW}[7/11] Conda/Mamba environments:${NC}"
if command -v conda &> /dev/null; then
    CONDA_ENV=$(conda info --envs 2>/dev/null | grep '*' | awk '{print $1}')
    echo -e "  ${GREEN}[√] Conda found (active: $CONDA_ENV)${NC}"
else
    echo -e "  ${YELLOW}[!] Conda not found${NC}"
fi
if command -v micromamba &> /dev/null; then
    MAMBA_VERSION=$(micromamba --version 2>/dev/null)
    echo -e "  ${GREEN}[√] Micromamba: $MAMBA_VERSION${NC}"
else
    echo -e "  ${YELLOW}[!] Micromamba not found${NC}"
fi

# 8. uv check
echo -e "\n${YELLOW}[8/11] UV package manager:${NC}"
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>/dev/null)
    echo -e "  ${GREEN}[√] $UV_VERSION${NC}"
else
    echo -e "  ${YELLOW}[!] UV not found (not installed)${NC}"
fi

# 9. HTTP/HTTPS/APT mirror check
echo -e "\n${YELLOW}[9/11] Network connectivity check:${NC}"
# Check basic connectivity
if curl -s -I --max-time 3 "https://www.google.com" -o /dev/null -w "%{http_code}" | grep -qE "200|301|302"; then
    echo -e "  ${GREEN}[√] Internet connectivity: OK${NC}"
else
    echo -e "  ${RED}[X] Internet connectivity: FAILED${NC}"
fi
# Check proxy status
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ]; then
    echo -e "  ${YELLOW}[!] Proxy detected: http_proxy=${http_proxy:-unset}, https_proxy=${https_proxy:-unset}${NC}"
else
    echo -e "  ${GREEN}[√] No proxy configured${NC}"
fi
# Check APT mirror
if [ -f /etc/apt/sources.list ]; then
    APT_MIRROR=$(grep -v "^#" /etc/apt/sources.list | head -1 | awk '{print $2}' | cut -d'/' -f3)
    echo -e "  ${GREEN}[√] APT mirror: $APT_MIRROR${NC}"
fi

# 10. Permission check (project directory)
echo -e "\n${YELLOW}[10/11] Permission checks:${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -w "$SCRIPT_DIR" ]; then
    echo -e "  ${GREEN}[√] Write permission: $SCRIPT_DIR${NC}"
else
    echo -e "  ${RED}[X] No write permission: $SCRIPT_DIR${NC}"
fi
# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "  ${YELLOW}[!] Running as root (use with caution)${NC}"
else
    echo -e "  ${GREEN}[√] Running as non-root user: $USER${NC}"
fi

# 11. Python interpreter check
echo -e "\n${YELLOW}[11/11] Python interpreter:${NC}"
if command -v python &> /dev/null; then
    PYTHON_PATH=$(which python)
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "  ${GREEN}[√] Python: $PYTHON_VERSION ($PYTHON_PATH)${NC}"
elif command -v python3 &> /dev/null; then
    PYTHON_PATH=$(which python3)
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "  ${GREEN}[√] Python3: $PYTHON_VERSION ($PYTHON_PATH)${NC}"
else
    echo -e "  ${RED}[X] Python not found!${NC}"
fi
# Check pip availability
if command -v pip &> /dev/null; then
    echo -e "  ${GREEN}[√] pip available${NC}"
fi


if [ -f ~/.jupyter/jupyter_server_config.py ]; then
    echo -e "\n${YELLOW}[Jupyter Config Check]:${NC}"
    grep "websocket_max_message_size" ~/.jupyter/jupyter_server_config.py || echo "  [!] Not found"
else
    echo -e "\n${YELLOW}[Jupyter Config Check]: Config file not found, skipping.${NC}"
fi


# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ System check completed!${NC}"
echo -e "${YELLOW}Tip: Run 'tail -f train.log' to monitor training progress${NC}"
echo -e "${BLUE}========================================${NC}"