# CIFAR-10 Image Classification with Custom CNN

This repository contains an implementation of a Convolutional Neural Network (CNN) built entirely from scratch in Python for the CIFAR-10 image classification task. The implementation includes all necessary components without relying on deep learning frameworks for the core neural network operations.

## Overview

The CIFAR-10 dataset consists of 60,000 RGB images (32x32 pixels) distributed across 10 categories. This project demonstrates how to build a CNN architecture that achieves over 82% accuracy on the test set using only basic numerical computing libraries.

## Features

- **Complete CNN Implementation**: All neural network components are implemented from scratch, including:
  - Convolutional layers with im2col optimization
  - MaxPooling layers
  - ReLU activation
  - Fully connected layers
  - Batch normalization
  - Dropout regularization
  
- **Advanced Training Techniques**:
  - One Cycle Learning Rate scheduling
  - Data augmentation (horizontal flipping and cutout)
  - Weight decay regularization
  - Learning rate finder

- **Performance Analysis**:
  - Confusion matrix visualization
  - Precision, recall, and F1-score metrics
  - Training/validation loss curves
  - Comparative analysis with other methods

## Architecture

The CNN follows a standard architecture with three convolutional blocks followed by fully connected layers:

1. **First Convolutional Block**
   - Conv Layer: 3 input channels, 32 output channels, 3x3 kernel, padding 1
   - BatchNorm
   - ReLU
   - MaxPool: 2x2 window, stride 2

2. **Second Convolutional Block**
   - Conv Layer: 32 input channels, 64 output channels, 3x3 kernel, padding 1
   - BatchNorm
   - ReLU
   - MaxPool: 2x2 window, stride 2

3. **Third Convolutional Block**
   - Conv Layer: 64 input channels, 128 output channels, 3x3 kernel, padding 1
   - BatchNorm
   - ReLU
   - MaxPool: 2x2 window, stride 2

4. **Fully Connected Layers**
   - Flatten
   - FC Layer: 128*4*4 → 256
   - ReLU
   - Dropout (50%)
   - FC Layer: 256 → 10 (output)

## Results

The model achieves:
- **Training Accuracy**: 86.62%
- **Test Accuracy**: 82.44%

Compared to simpler models tested:
- Nearest Neighbor (k=1): 35.39%
- Nearest Neighbor (k=3): 33.03%
- Nearest Class Centroid: 27.74%

## Requirements

- Python 3.x
- NumPy
- CuPy (for GPU acceleration)
- Matplotlib (for visualization)
- tqdm (for progress bars)

## Usage

```python
# Train the model
model = CNNModel()
loss_history, test_loss_history = train_model(model, X_train, y_train, X_test, y_test, 
                                             num_epochs=70, batch_size=64, 
                                             learning_rate=0.8, reg_lambda=0.005)

# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Visualize results
plot_loss_curves(loss_history, test_loss_history)
plot_confusion_matrix(model, X_test, y_test)
plot_performance_metrics(model, X_test, y_test)
```

## Implementation Details

The repository includes custom implementations of:

1. **Helper Functions**:
   - `im2col_indices` and `col2im_indices` for efficient convolution operations
   - `softmax` and `cross_entropy_loss` for classification
   - Data augmentation functions like `random_horizontal_flip` and `cutout`

2. **Network Layers**:
   - `ConvLayer`: Implements convolutional operations
   - `MaxPool`: Performs max pooling
   - `ReLU`: Applies ReLU activation
   - `FCLayer`: Implements fully connected layers
   - `BatchNorm`: Applies batch normalization
   - `Dropout`: Implements dropout regularization

3. **Optimization**:
   - `OneCycleLR`: Implements the one-cycle learning rate policy

## Future Work

- Implement more advanced architectures (ResNet, DenseNet)
- Add more data augmentation techniques
- Experiment with different optimization strategies
- Support for more datasets

## Acknowledgements

This project was completed as part of the "Neural Networks - Deep Learning" course. The implementation demonstrates a deep understanding of the underlying principles of convolutional neural networks by building every component from the ground up.