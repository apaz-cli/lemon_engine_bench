#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing all engines..."

# Detect GPU architecture
if command -v rocminfo &> /dev/null; then
    ARCH=$(rocminfo | grep -oP 'Name:\s*\K\w+' | head -1)
    echo "Detected GPU architecture: $ARCH"
    if [[ "$ARCH" == "gfx1100" ]]; then
        echo "RDNA3 (gfx1100) detected. SGLang may not support this architecture."
        export SKIP_SGLANG=1
    fi
else
    echo "rocminfo not found. Assuming ROCm is not installed."
fi

# Install each engine
for script in install_*.sh; do
    if [[ "$script" == "install_all.sh" ]]; then
        continue
    fi
    if [[ "$script" == "install_sglang.sh" && -n "$SKIP_SGLANG" ]]; then
        echo "Skipping $script due to RDNA3"
        continue
    fi
    echo "Running $script..."
    bash "$script"
done

echo "All installations complete."