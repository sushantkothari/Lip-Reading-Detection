# Lip Reading Detection

This project implements a deep learning model for lip reading detection using TensorFlow and Keras. The model is trained to predict spoken words from video frames of lip movements.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Lip reading detection is a challenging task in computer vision and speech recognition. This project aims to develop a deep learning model that can accurately predict spoken words by analyzing the lip movements of a speaker in video frames. The model utilizes a combination of 3D convolutional neural networks (CNNs) and bidirectional long short-term memory (LSTM) layers to capture both spatial and temporal features of lip movements.

## Features

- Video frame preprocessing and data augmentation
- Custom data pipeline for efficient loading and batching
- Deep neural network architecture for lip reading detection
- Training with CTC (Connectionist Temporal Classification) loss
- Evaluation metrics including accuracy, precision, recall, and F1 score
- Visualization of training progress and results

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Matplotlib
- NumPy
- ImageIO

## Installation

1. Clone the repository:
```
git clone https://github.com/sushantkothari/lip-reading-detection.git
cd lip-reading-detection
```
2. Install the required dependencies:
```
pip install tensorflow opencv matplotlib numpy imagio
```

## Usage

1. Prepare your dataset and update the data loading functions in the script.
2. Run the training ipynb:
```
Lip_Reading_Detection.ipynb
```
3. For prediction on new videos, use the `load_data` and `predict` functions from the trained model.

## Data Pipeline

The data pipeline includes the following steps:
1. Loading video frames and converting them to grayscale
2. Preprocessing frames by cropping and normalizing
3. Loading alignment data for word labels
4. Creating a TensorFlow dataset with shuffling, batching, and prefetching

## Model Architecture

The model architecture consists of:
1. 3D Convolutional layers for spatial feature extraction
2. MaxPooling layers for downsampling
3. Bidirectional LSTM layers for temporal feature extraction
4. Dense layer for final prediction

## Training

The model is trained using:
- CTC loss function
- Adam optimizer
- Learning rate scheduling
- Custom callbacks for checkpointing and example generation

## Evaluation Metrics

The model's performance is evaluated using:
- Accuracy: Percentage of correctly predicted words
- Precision: Ratio of correctly predicted positive observations to total predicted positive observations
- Recall: Ratio of correctly predicted positive observations to all actual positive observations
- F1 Score: Harmonic mean of precision and recall

## Future Improvements

- Experiment with different model architectures
- Incorporate attention mechanisms
- Explore transfer learning from pre-trained models
- Increase dataset size and diversity
- Implement real-time lip reading for live video streams

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
