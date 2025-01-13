# Dataset Structure for Emotion Recognition

This project uses two datasets for training the emotion recognition models:

1. RAVDESS Dataset
2. CREMA-D Dataset

## Directory Structure

```
datasets/
├── RAVDESS/
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ...
└── CREMA-D/
    ├── AudioWAV/
    ├── VideoFlash/
    └── VideoDemographics.csv
```

## Dataset Setup Instructions

### RAVDESS Dataset
1. Download the RAVDESS dataset from https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
2. Extract the files
3. Place the Actor_* folders in the `datasets/RAVDESS/` directory

### CREMA-D Dataset
1. Download the CREMA-D dataset from https://github.com/CheyneyComputerScience/CREMA-D
2. Extract the files
3. Place the following in the `datasets/CREMA-D/` directory:
   - AudioWAV folder containing WAV files
   - VideoFlash folder containing video files
   - VideoDemographics.csv file

## Training Process

1. Ensure both datasets are properly placed in their respective directories
2. Run the training script:
   ```bash
   cd server
   python train.py
   ```

3. Monitor the training progress in the terminal
4. View training metrics and visualizations in the `server/metrics/` directory

## Metrics and Visualization

After training, the following metrics will be generated in `server/metrics/`:

- Confusion matrices
- Classification reports
- Training curves
- Model performance comparisons

## Model Files

Trained models will be saved in:
- `server/models/audio_model.pth`
- `server/models/video_model.pth`