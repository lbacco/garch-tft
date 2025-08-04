#!/bin/bash

# The path to your Python script
PYTHON_SCRIPT_PATH="./anngarch.py"
# Step 1: Setup environment.
echo
echo Setting up virtual environment...
echo
OUTPUT_FOLDER="./TUNER"
set -e


#pip3.8 install virtualenv # Assumes pip3 is installed!
#python3.8 -m virtualenv $OUTPUT_FOLDER/venvmod
source $OUTPUT_FOLDER/venvmod/bin/activate
pip3.8 install -r requirementslstm.txt

# Configurazione di CUDA
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Configurazione di TensorRT e cuDNN
export LD_LIBRARY_PATH=$OUTPUT_FOLDER/venvmod/lib/python3.8/site-packages/tensorrt:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$OUTPUT_FOLDER/venvmod/lib/python3.8/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$OUTPUT_FOLDER/venvmod/lib/python3.8/site-packages/nvidia/cudnn/include:${LD_LIBRARY_PATH}

export CUDA_VISIBLE_DEVICES='1'
python3.8 $PYTHON_SCRIPT_PATH

# Deactivate your virtual environment if you used one
deactivate

