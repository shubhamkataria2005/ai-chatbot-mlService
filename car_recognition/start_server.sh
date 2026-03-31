#!/bin/bash

echo "Starting Car Recognition ML Server..."
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Install requirements
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Start the server
echo "Starting Flask server on port 5002..."
python3 server.py