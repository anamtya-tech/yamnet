#!/bin/bash
# Build script for YAMNet ZODAS integration example

set -e

TENSORFLOW_ROOT="/home/azureuser/tensorflow"
TFLITE_LIB="${TENSORFLOW_ROOT}/bazel-bin/tensorflow/lite"
FLATBUFFERS_INC="${TENSORFLOW_ROOT}/bazel-tensorflow/external/flatbuffers/include"

echo "=========================================="
echo "Building ZODAS Integration Example"
echo "=========================================="
echo ""

# Check files
if [ ! -f "yamnet_classifier.h" ]; then
    echo "Error: yamnet_classifier.h not found"
    exit 1
fi

if [ ! -f "yamnet_classifier.cpp" ]; then
    echo "Error: yamnet_classifier.cpp not found"
    exit 1
fi

if [ ! -f "zodas_integration_example.cpp" ]; then
    echo "Error: zodas_integration_example.cpp not found"
    exit 1
fi

echo "Building zodas_integration_example..."
g++ -std=c++17 -O3 \
  -I${TENSORFLOW_ROOT} \
  -I${FLATBUFFERS_INC} \
  zodas_integration_example.cpp \
  yamnet_classifier.cpp \
  -L${TFLITE_LIB} \
  -ltensorflowlite \
  -ldl -lpthread \
  -o zodas_example

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "Executable: ./zodas_example"
    echo ""
    echo "To run:"
    echo "  export LD_LIBRARY_PATH=${TFLITE_LIB}:\$LD_LIBRARY_PATH"
    echo "  ./zodas_example"
    echo ""
    echo "Or use the run script:"
    echo "  LD_LIBRARY_PATH=${TFLITE_LIB}:\$LD_LIBRARY_PATH ./zodas_example"
    echo ""
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi