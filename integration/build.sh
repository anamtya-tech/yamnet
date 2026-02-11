#!/bin/bash
# Build script for YAMNet Core C++ Classifier

set -e  # Exit on error

# Configuration
TENSORFLOW_ROOT="/home/azureuser/tensorflow"
TFLITE_LIB="${TENSORFLOW_ROOT}/bazel-bin/tensorflow/lite"
FLATBUFFERS_INC="${TENSORFLOW_ROOT}/bazel-tensorflow/external/flatbuffers/include"

echo "========================================"
echo "Building YAMNet Core C++ Classifier"
echo "========================================"
echo ""

# Check if TensorFlow directory exists
if [ ! -d "$TENSORFLOW_ROOT" ]; then
    echo "Error: TensorFlow directory not found: $TENSORFLOW_ROOT"
    exit 1
fi

# Check if TFLite library exists
if [ ! -f "${TFLITE_LIB}/libtensorflowlite.so" ] && [ ! -f "${TFLITE_LIB}/libtensorflowlite.a" ]; then
    echo "Error: TensorFlow Lite library not found in: $TFLITE_LIB"
    echo "Make sure you've built TensorFlow Lite:"
    echo "  cd $TENSORFLOW_ROOT"
    echo "  bazel build -c opt //tensorflow/lite:libtensorflowlite.so"
    exit 1
fi

# Compile
echo "Compiling yamnet_core_classifier.cpp..."
g++ -std=c++17 -O3 \
  -I${TENSORFLOW_ROOT} \
  -I${FLATBUFFERS_INC} \
  yamnet_core_classifier.cpp \
  -L${TFLITE_LIB} \
  -ltensorflowlite \
  -ldl -lpthread \
  -o yamnet_core_classifier

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "Executable: ./yamnet_core"
    echo ""
    echo "Usage:"
    echo "  ./yamnet_core <model.tflite> <class_map.csv> <audio.wav>"
    echo ""
    echo "Example:"
    echo "  ./yamnet_core yamnet_core.tflite yamnet_class_map.csv wavs/miaow_16k.wav"
    echo ""
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi