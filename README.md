# Satellite-sensory-image-classification
Satellite imagery enables the monitoring and assessment of environmental changes and disaster management. The dataset has 4 classes cloudy, desert, green_area and water.
1. Introduction

The task is to classify satellite images of 4 classes with new data using deep learning model. Implemented simple CNN but the model was overfitting. So used pre-trained resnet-18 model by using transfer learning. Hyper parameter tuning was performed to find the optimal hyperparameters. And got the accuracy of 97.71 %.

2. Dataset Description

The dataset can be downloaded from https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification/data
It is a satellite sensory images of 4 classes- clody, desert, green_area, water
The size of the entire dataset: 5631 which is in jpg mode.
Ech class has size of 1500 except desert class with the size of 1131

3. Model Architecture

CNN:
The convolutional layers have 16, 32, and 64 filters respectively, each with a kernel size of 3x3 and padding of 1. These layers learn various feature representations from the input images. The first fully connected layer has 64 * 28 * 28 input features, which corresponds to the flattened output of the last convolutional layer. Applied dropout(0.5) in between the two fully connected layers.
Resnet18:
Resnet18 pre-trained model consists of 18 layers, including convolutional layers, pooling layers, and residual blocks. It uses residual connections to enable the training of deep neural networks while mitigating the vanishing gradient problem. freezes the parameters of the pre-trained ResNet-18 model to prevent them from being updated during training. The fully connected layer (the last layer) of the ResNet-18 model is replaced with a new linear layer having 4 output classes. 

4. Training process:

CNN:
Optimizer: Utilized Stochastic Gradient Descent (SGD) for updating model parameters based on loss gradients.
Loss Function: Employed CrossEntropyLoss() suitable for multi-class classification tasks.
Batch Size and Epochs: Set batch size to 32 and trained for 10 epochs.
Learning Rate: Set at 0.01 to balance convergence speed and stability during optimization.
Weight Decay: Applied a regularization term of 0.01 to penalize large parameter values, preventing overfitting.
Training Metrics: Calculated training loss and accuracy after each epoch using batch mean.
Validation Process: Evaluated model performance on validation data, computing average loss and accuracy.Testing Process: Test model performance on unseen data, computing accuracy

Resnet18 Model:
Optimizer: Utilized Stochastic Gradient Descent (SGD) for updating model parameters based on loss gradients.
Loss Function: Employed CrossEntropyLoss() suitable for multi-class classification tasks.
Batch Size and Epochs: Set batch size to 32 and trained for 10 epochs.
Learning Rate: Set at 0.01 to balance convergence speed and stability during optimization.
Weight Decay: Applied a regularization term of 0.01 to penalize large parameter values, preventing overfitting.
Hyperparameter Tuning: Tuned learning rate to 0.01 and weight decay to 0.01 for optimal performance.
Training Metrics: Calculated training loss and accuracy after each epoch using batch mean.
Validation Process: Evaluated model performance on validation data, computing average loss and accuracy.
Testing Process: Test model performance on unseen data, computing accuracy
Early Stopping: Implemented early stopping if validation loss did not improve for a certain number of epochs (patience).
Convergence Monitoring: Updated best validation loss and epochs without improvement counters to track model convergence.
Overall Goal: Aimed to effectively train the model while promoting generalization and convergence.

5. Evaluation of the Resnet18 model:

Training Loss: 0.06889394031216702, Training Accuracy: 0.9815277777777778, Validation Accuracy: 0.9829166666666667, Validation Loss: 0.06296364814043046
Accuracy of the test data: 97.71%
Precision: 0.9773726833887795
Recall: 0.9770833333333333
F1 Score: 0.9770709468405194

6. Test the model:
The model predicted correct for unseen images as cloudy, desert and water except green area.

7. How to Run the code:
First, download the folder from Kaggle, then compress it into a zip file, extract its contents, and execute the code. The URLs of the images for testing the model are specified in the notebook; make the necessary adjustments accordingly.
