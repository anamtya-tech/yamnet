#!/bin/bash
# Build script for YAMNet API test

set -e

TENSORFLOW_ROOT="/home/azureuser/tensorflow"
TFLITE_LIB="${TENSORFLOW_ROOT}/bazel-bin/tensorflow/lite"
FLATBUFFERS_INC="${TENSORFLOW_ROOT}/bazel-tensorflow/external/flatbuffers/include"

echo "=========================================="
echo "Building YAMNet API Test"
echo "=========================================="
echo ""

echo "Building test_yamnet_api..."
g++ -std=c++17 -O3 \
  -I${TENSORFLOW_ROOT} \
  -I${FLATBUFFERS_INC} \
  test_yamnet_api.cpp \
  yamnet_classifier.cpp \
  -L${TFLITE_LIB} \
  -ltensorflowlite \
  -ldl -lpthread \
  -o test_yamnet_api

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "Run the test:"
    echo "  LD_LIBRARY_PATH=${TFLITE_LIB}:\$LD_LIBRARY_PATH ./test_yamnet_api"
    echo ""
    echo "Or with custom audio:"
    echo "  LD_LIBRARY_PATH=${TFLITE_LIB}:\$LD_LIBRARY_PATH ./test_yamnet_api path/to/audio.wav"
    echo ""
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi