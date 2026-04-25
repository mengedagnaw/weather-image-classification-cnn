# weather-image-classification-cnn
CNN-based image classification model for four weather categories: cloudy, rain, shine, and sunrise.
# Weather Image Classification Using CNN

This repository contains a convolutional neural network implementation for a four-class weather image classification task. The model classifies weather images into the following categories:

- cloudy
- rain
- shine
- sunrise

The project is implemented using TensorFlow/Keras and scikit-learn.

## Project Description

The objective of this project is to train and evaluate a convolutional neural network for weather image classification using a 75/25 train-test split.

The workflow includes:

- detecting image classes from folder names
- preprocessing and resizing images
- applying stratified train-test splitting
- training a CNN model
- evaluating model performance using accuracy, balanced accuracy, Cohen's kappa, confusion matrix, and classification report

## Repository Contents

```text
.
├── README.md
├── weather_cnn_task.py
└── weather_cnn_task.ipynb
```

## Dataset

The dataset contains four weather classes:

| Class | Description |
|---|---|
| cloudy | Cloudy weather images |
| rain | Rainy weather images |
| shine | Sunny or shining weather images |
| sunrise | Sunrise weather images |

The code expects the dataset to be organized with one folder per class.

## Methodology

A stratified 75/25 train-test split is used to preserve the class distribution in both training and test sets.

The CNN model uses:

- image resizing to 128 × 128 pixels
- random horizontal flipping
- random rotation
- random zoom
- convolutional layers
- max-pooling layers
- dropout regularization
- dense classification layers
- softmax output for four-class classification

## Model Architecture

The CNN consists of four convolutional blocks followed by a dense classification head:

```text
Input image: 128 × 128 × 3

Conv2D → MaxPooling
Conv2D → MaxPooling
Conv2D → MaxPooling
Conv2D → MaxPooling
Dropout
Flatten
Dense
Dropout
Dense Softmax Output
```

## Evaluation Metrics

The model is evaluated using:

- accuracy
- balanced accuracy
- Cohen's kappa
- confusion matrix
- precision
- recall
- F1-score

These metrics provide both overall and class-specific performance information.

## Requirements

Install the required Python packages using:

```bash
pip install tensorflow scikit-learn matplotlib pandas numpy pillow
```

Alternatively, create a `requirements.txt` file containing:

```text
tensorflow
scikit-learn
matplotlib
pandas
numpy
pillow
```

Then install with:

```bash
pip install -r requirements.txt
```

## How to Run

Run the Python script:

```bash
python weather_cnn_task1.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook weather_cnn_task1.ipynb
```

## Output

The code reports:

- detected class names
- number of images per class
- train-test split counts
- model training curves
- test accuracy
- balanced accuracy
- Cohen's kappa
- confusion matrix
- classification report

## Reproducibility

A fixed random seed is used:

```python
SEED = 42
```

This improves reproducibility for the train-test split, TensorFlow initialization, and dataset shuffling.

## Methodological Note

The assignment requires a 75/25 train-test split. In this implementation, the held-out 25% subset is also used as validation data during training.

This is acceptable for a coursework baseline, but a more rigorous machine learning workflow would use one of the following:

- separate train, validation, and test sets
- repeated stratified holdout
- k-fold cross-validation

## Author

Prepared for a weather image classification task using convolutional neural networks.

## License

This repository is intended for academic and educational use.
