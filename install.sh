#!/bin/bash

set -e

PYTHON=${PYTHON:-python3.12}

# Verify Python 3.12
if ! $PYTHON -c "import sys; assert sys.version_info[:2] == (3, 12)"; then
    echo "Error: Python 3.12 is required."
    exit 1
fi

#sudo apt-get update
python3.12 -m pip install --upgrade pip setuptools wheel
python3.12 -m pip install -r requirements/base.txt
case "$1" in
    mac)
        python3.12 -m pip install -r requirements/mac.txt
        ;;
    colab)
        python3.12 -m pip install -r requirements/colab.txt
        ;;
    *)
        echo "Usage: ./setup.sh [mac|colab]"
        exit 1
        ;;
esac