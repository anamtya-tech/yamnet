#!/bin/bash
# Run YAMNet API test with proper library path

TFLITE_LIB="/home/azureuser/tensorflow/bazel-bin/tensorflow/lite"

WAV_FILE="${1:-wavs/miaow_16k.wav}"

echo "Testing with: $WAV_FILE"
echo ""

LD_LIBRARY_PATH="${TFLITE_LIB}:${LD_LIBRARY_PATH}" ./test_yamnet_api "$WAV_FILE"