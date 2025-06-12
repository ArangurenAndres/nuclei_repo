#!/bin/bash

# setup.sh
# This script sets up the Python environment and runs the main application.

echo "--- Starting setup process ---"

# 1. Create a virtual environment (optional but highly recommended)
echo "Creating and activating virtual environment..."
python3 -m venv venv
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to create or activate virtual environment."
    exit 1
fi
echo "Virtual environment activated."

# 2. Install Python dependencies from requirements.txt
echo "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies from requirements.txt."
    echo "Please check requirements.txt and your internet connection."
    deactivate # Deactivate venv on error
    exit 1
fi
echo "All dependencies installed successfully."

# 3. Run the main application (run.py)
echo "Running the main application (run.py)..."
python run.py
if [ $? -ne 0 ]; then
    echo "Error: run.py exited with an error."
    deactivate # Deactivate venv on error
    exit 1
fi
echo "run.py executed successfully."

echo "--- Setup process complete ---"
echo "To deactivate the virtual environment, run 'deactivate'."