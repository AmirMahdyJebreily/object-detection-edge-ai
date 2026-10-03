#!/bin/bash
set -e

echo "Compiling all Python files with python3.5..."
python3.5 -m py_compile *.py adapters/*.py core/*.py utils/*.py

echo "Checking syntax successfully completed."

# If there are unit tests, we could run them here, e.g.:
# python3.5 -m unittest discover tests

echo "CI completed successfully!"

