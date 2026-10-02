#!/bin/bash

# Exit on any error to prevent cascading failures
set -e

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
if [ ! -d "venv" ]; then
    # CRITICAL: --system-site-packages allows the venv to use python3-numpy, opencv and psutil from apt!
    python3 -m venv --system-site-packages venv || echo "Warning: python3-venv might not be supported on this old version. Continuing without it..."
fi
# Activate venv if it was successfully created
[ -f "venv/bin/activate" ] && source venv/bin/activate

echo "[4/4] Installing Python requirements in venv..."
PYTHON_VERSION=$(python3 -c 'import sys; print(str(sys.version_info.major) + "." + str(sys.version_info.minor))')
ARCH=$(uname -m)

if [ "$PYTHON_VERSION" = "3.5" ] && [ "$ARCH" = "armv7l" ]; then
    echo "Detected Python 3.5 on armv7l (NanoPi)."
    echo "Installing pre-built tflite_runtime wheel..."
    pip install https://github.com/google-coral/pycoral/releases/download/v1.0.1/tflite_runtime-2.5.0-cp35-cp35m-linux_armv7l.whl
    
    # Try to get OpenCV from piwheels (compiled for 32-bit ARM) in case apt failed
    pip install opencv-python --extra-index-url https://www.piwheels.org/simple || echo "OpenCV pip install failed, hoping apt worked..."
    
    grep -vE 'opencv-python|tflite-runtime' requirements.txt > requirements_filtered.txt
    pip install -r requirements_filtered.txt --extra-index-url https://www.piwheels.org/simple || echo "Note: Some extra UI pip packages might fail on Py3.5."
    rm requirements_filtered.txt
else
    echo "Installing tensorflow-cpu and dependencies via pip..."
    # On desktop/modern systems, tflite_runtime may not be available for the newest Python versions (e.g. 3.14)
    # We use tensorflow-cpu as it provides tf.lite which our adapter uses as fallback.
    pip install tensorflow-cpu
    grep -vE 'opencv-python' requirements.txt > requirements_filtered.txt
    pip install -r requirements_filtered.txt
    rm requirements_filtered.txt
fi

echo "===================================================="
echo "Setup complete! You can now run the server:"
echo "source venv/bin/activate"
echo "python main.py"
echo "===================================================="

