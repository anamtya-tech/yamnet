#!/bin/bash
# Run script for YAMNet Core C++ Classifier
# Automatically sets LD_LIBRARY_PATH

# TensorFlow Lite library path
TFLITE_LIB="/home/azureuser/tensorflow/bazel-bin/tensorflow/lite"

# Set library path and run
LD_LIBRARY_PATH="${TFLITE_LIB}:${LD_LIBRARY_PATH}" ./yamnet_core_classifier "$@"