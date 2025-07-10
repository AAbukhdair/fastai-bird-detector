# FastAI Bird Detector

A real-time bird species classifier using FastAI and OpenCV on macOS.

## Features

- Fine-tunes a ResNet34 on the CUB-200 dataset (200 species)
- Live webcam inference with a 50% confidence threshold
- Quick setup via `train_export.py` and `detect_birds.py`

## Prerequisites

- Python 3.8+  
- A virtual environment (venv or conda)  
- A webcam with camera permission granted to your Terminal

## Installation

```bash
# 1. Clone & cd into repo
git clone https://github.com/yourusername/fastai-bird-detector.git
cd fastai-bird-detector

# 2. Create & activate venv
python3 -m venv birdenv
source birdenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

