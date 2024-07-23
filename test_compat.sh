#!/bin/bash

# Set the environment name
ENV_NAME="test-tflearn-compat"

# Remove the environment if it exists
mamba env remove -n $ENV_NAME -y

# Create the Conda environment with Python 3.10
mamba create -n $ENV_NAME python=3.10 -y

# Activate the environment
source activate $ENV_NAME

# Install specific versions of TensorFlow and related packages
pip install tensorflow-macos==2.9.0
pip install tensorflow-metal==0.6.0
pip install numpy==1.24.3
pip install h5py==3.8.0
pip install Pillow==8.4.0

# Install TFLearn from local repo
pip install -e .

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import tensorflow.compat.v1 as tf; print('TensorFlow compat.v1 imported successfully')"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

# Run the existing test script
python tflearn/test_compat.py
