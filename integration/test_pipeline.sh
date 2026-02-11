#!/bin/bash
# End-to-end test script using Python FFT + C++ mel/inference

set -e

WAV_FILE="${1:-wavs/miaow_16k.wav}"
SPECTRA_FILE="spectra.bin"

echo "=========================================="
echo "YAMNet Core Test (Python FFT + C++ Pipeline)"
echo "=========================================="
echo ""

# Step 1: Compute FFT using Python
echo "Step 1: Computing STFT magnitudes (Python)..."
python compute_fft.py "$WAV_FILE" "$SPECTRA_FILE"
echo ""

# Step 2: Process with C++ (mel + inference)
echo "Step 2: Computing mel + running inference (C++)..."
LD_LIBRARY_PATH=/home/azureuser/tensorflow/bazel-bin/tensorflow/lite:$LD_LIBRARY_PATH \
  ./yamnet_core_classifier yamnet_core.tflite yamnet_class_map.csv --spectra "$SPECTRA_FILE"

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo ""
echo "This simulates ZODAS integration:"
echo "  - Python FFT  → ZODAS will provide 257-bin spectra"
echo "  - C++ mel+inference → Your production code"