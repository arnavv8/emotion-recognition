#!/bin/bash

# Create necessary directories
mkdir -p metrics
mkdir -p models

# Install requirements
pip install -r requirements.txt

# Run the interactive training script
python train.py

echo "Training complete! Check metrics directory for results."