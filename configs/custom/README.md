## Configurations adopt with our own defined model
Here is the config file for this experiment.

It starts with data loading and data processing, and then configures the backbone, neck, and head of the model, as well as the model preprocessing and testing configurations. The basic configuration for training and validation is completed, as well as log processing and visualisation. This configuration file provides a comprehensive framework for model training and validation, covering all steps from data processing to model training and validation. It enables the user to define and manage all settings and parameters in a structured and modular way.

In this experiment, we tested the accuracy of different backbone with different loss functions at different epochs.

### AlexNet
- `topd-Alexnet-20epochs.py`
- `topd-Alexnet-50epochs.py`
- `topd-Alexnet-RLELoss-20epochs.py`
- `topd-Alexnet-RLELoss-50epochs.py`
  
### ResNet
- `topd-Resnet50-20epochs.py` 
- `topd-Resnet50-RLELoss-20epochs.py` 
- `topd-Resnet50-RLELoss-100epochs.py`
- `topd-Resnet101-20epochs.py` 
- `topd-Resnet101-RLELoss-100epochs.py` 
- `topd-Resnet152-20epochs.py` 
-  `topd-Resnet152-RLELoss-20epochs.py` 

### SCNet
- `topd-SCnet101-20epochs.py`
- `topd-SCnet101-RLELoss-50epochs.py`
- `topd-SCnet101-RLELoss-100epochs.py`
- `topd-SCnet101-RLELoss-20epochs.py`



