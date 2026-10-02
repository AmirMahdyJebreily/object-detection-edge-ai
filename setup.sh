#!/bin/bash

# Exit on any error to prevent cascading failures
set -e

# Fix for "locale.Error: unsupported locale setting" on minimal Linux installations
export LC_ALL=C
export LANG=C

echo "===================================================="
echo "  Setting up Conveyor Belt Object Detection on Edge "
echo "===================================================="

echo "[1/4] Updating system packages..."
# Ensure universe repository is enabled (required for python3-opencv on Ubuntu 16.04)
sudo apt-get install -y software-properties-common || true
sudo add-apt-repository universe || true
sudo apt update

echo "[2/4] Installing system dependencies..."
sudo apt install -y python3-numpy python3-pip python3-venv python3-psutil wget || true
sudo apt install -y python3-opencv || echo "Warning: python3-opencv not found in apt. We will attempt to install it via pip/piwheels."

echo "[3/4] Setting up Virtual Environment (venv)..."
PYTHON_VERSION=$(python3 -c 'import sys; print(str(sys.version_info.major) + "." + str(sys.version_info.minor))')
ARCH=$(uname -m)

if [ "$PYTHON_VERSION" = "3.5" ] && [ "$ARCH" = "armv7l" ]; then
    echo "Skipping venv creation on NanoPi (Python 3.5) to avoid ensurepip errors."
    echo "We will install packages globally using --user."
    PIP_CMD="python3 -m pip install --user"
else
    if [ ! -d "venv" ]; then
        python3 -m venv --system-site-packages venv || true
    fi
    [ -f "venv/bin/activate" ] && source venv/bin/activate
    PIP_CMD="pip install"
fi

echo "[4/4] Installing Python requirements..."

if [ "$PYTHON_VERSION" = "3.5" ] && [ "$ARCH" = "armv7l" ]; then
    echo "Detected Python 3.5 on armv7l (NanoPi)."
    echo "Installing pre-built tflite_runtime wheel..."
    $PIP_CMD https://github.com/google-coral/pycoral/releases/download/v1.0.1/tflite_runtime-2.5.0-cp35-cp35m-linux_armv7l.whl
    
    # Try to get OpenCV from piwheels (compiled for 32-bit ARM) in case apt failed
    $PIP_CMD opencv-python --extra-index-url https://www.piwheels.org/simple || echo "OpenCV pip install failed, hoping apt worked..."
    
    grep -vE 'opencv-python|tflite-runtime' requirements.txt > requirements_filtered.txt
    $PIP_CMD -r requirements_filtered.txt --extra-index-url https://www.piwheels.org/simple || echo "Note: Some extra UI pip packages might fail on Py3.5."
    rm requirements_filtered.txt
else
    echo "Installing tensorflow-cpu and dependencies via pip..."
    # On desktop/modern systems, tflite_runtime may not be available for the newest Python versions (e.g. 3.14)
    # We use tensorflow-cpu as it provides tf.lite which our adapter uses as fallback.
    $PIP_CMD tensorflow-cpu
    grep -vE 'opencv-python' requirements.txt > requirements_filtered.txt
    $PIP_CMD -r requirements_filtered.txt
    rm requirements_filtered.txt
fi

echo "===================================================="
echo "Setup complete! You can now run the server:"
echo "source venv/bin/activate"
echo "python main.py"
echo "===================================================="

