#!/bin/bash

# Check if datasets exist
if [ ! -d "../datasets/RAVDESS" ] || [ ! -d "../datasets/CREMA-D" ]; then
    echo "Error: Datasets not found!"
    echo "Please ensure both RAVDESS and CREMA-D datasets are properly placed in the datasets directory."
    echo "Check datasets/README.md for setup instructions."
    exit 1
fi

# Create necessary directories
mkdir -p metrics
mkdir -p models

# Install requirements
pip install -r requirements.txt

# Run training
echo "Starting model training..."
python train.py

echo "Training complete! Check metrics directory for results."