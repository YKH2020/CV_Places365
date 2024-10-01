# Image Classification with ResNet50 (Pretrained) - 5 Scene Classes

This project implements an image classification model comparing the performance of the following pretrained architectures:
- ResNet50
- VGG16
- EfficientNet B2 (from geffnet)
- AlexNet

The model is trained on the Places365 dataset considering only 5 classes:

- `airport_terminal`
- `aquarium`
- `beach`
- `bar`
- `music_studio`

## Model Architecture

### ResNet50 Architecture:
ResNet50 is a deep residual network consisting of 50 layers. It is designed to solve the vanishing gradient problem by using skip connections (residual connections) that allow gradients to flow directly through the network, improving training and performance.
- **Key features**:
  - 49 convolutional layers
  - 1 fully connected (fc) layer
  - Skip (residual) connections that bypass one or more layers
  - Pretrained on ImageNet for feature extraction
  - Adapted by adding a custom classification head for 5 scene classes:
    - Linear(2048, 512) → ReLU
    - Dropout(0.6)
    - Linear(512, 256) → ReLU
    - Dropout(0.5)
    - Linear(256, 128) → ReLU
    - Dropout(0.4)
    - Linear(128, 5) (5 output classes)

### VGG16 Architecture:
VGG16 is a deep convolutional neural network with 16 layers, known for its simplicity and uniformity in using small (3x3) convolution filters and consistent max-pooling layers.
- **Key features**:
  - 13 convolutional layers followed by 3 fully connected layers
  - Uses max-pooling after each block of convolution layers
  - Pretrained on ImageNet
  - Adapted by adding a custom classification head for 5 scene classes:
    - Flatten layer
    - Fully connected layers added after the final max-pooling layer for classification

### EfficientNet B2 (from geffnet) Architecture:
EfficientNet is known for its balance between model accuracy and computational efficiency. The B2 variant has fewer layers compared to larger EfficientNet variants but offers an optimal balance of parameters and performance.
- **Key features**:
  - 9 blocks of convolution layers with squeeze-and-excitation optimization
  - Swish activation function
  - Pretrained on ImageNet
  - Adapted with a classification head for 5 scene classes:
    - Linear layer tailored to the dataset size with ReLU activations and dropout

### AlexNet Architecture:
AlexNet is one of the pioneering deep convolutional neural networks, with fewer layers and complexity compared to more modern architectures like ResNet and VGG.
- **Key features**:
  - 5 convolutional layers followed by 3 fully connected layers
  - Pretrained on ImageNet
  - Adapted with a custom classifier for 5 scene classes:
    - Final fully connected layers replaced to accommodate 5 output classes

## Model Configuration:
- **Pretrained models**: ResNet50, VGG16, EfficientNet B2, AlexNet (all pretrained on ImageNet)
- **New fully connected layers**:
  - ResNet50: 
    - Linear(2048, 512) → ReLU
    - Dropout(0.6)
    - Linear(512, 256) → ReLU
    - Dropout(0.5)
    - Linear(256, 128) → ReLU
    - Dropout(0.4)
    - Linear(128, 5)
  - Other models:
    - Similar architecture adapted to each model with layers added after feature extraction
- **Loss function**: Cross Entropy Loss
- **Optimizer**: Adam (with learning rate `1e-6`), optimizing only the new fully connected layers


Ref:
https://medium.com/@deeprodge/multi-class-image-classifier-using-pytorch-and-transfer-learning-1d8f5c8782c7
