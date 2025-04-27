#!/bin/bash
# CREATE A LOCAL ENV USING PYTHON 3.11
MYVENV="venv"

# Check if python3.11 exists
if command -v python3.11 &> /dev/null
then
    echo "Creating virtual environment using python3.11..."
    # Use python3.11 to create the virtual environment
    python3.11 -m venv $MYVENV
else
    echo "Error: python3.11 command not found. Cannot create venv."
    exit 1
fi

# Activate the new environment
echo "Activating the virtual environment..."
source $MYVENV/bin/activate

# Install dependencies (using the pip from the venv)
echo "Installing requirements..."
pip install -r requirements.txt

echo "Virtual environment '$MYVENV' created and activated with Python 3.11."
echo "Requirements installed."
