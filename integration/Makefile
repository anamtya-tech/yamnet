# Makefile for YAMNet Core C++ Classifier

# Paths - adjust these to your setup
TENSORFLOW_ROOT = /home/azureuser/tensorflow
TFLITE_LIB_DIR = $(TENSORFLOW_ROOT)/bazel-bin/tensorflow/lite

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall
INCLUDES = -I$(TENSORFLOW_ROOT) \
           -I$(TENSORFLOW_ROOT)/bazel-tensorflow/external/flatbuffers/include

LDFLAGS = -L$(TFLITE_LIB_DIR)
LIBS = -ltensorflowlite -ldl -lpthread

# Targets
TARGET = yamnet_core
SOURCE = yamnet_core_cpp.cpp

# Build rules
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) $(LDFLAGS) $(LIBS) -o $(TARGET)
	@echo ""
	@echo "Build successful!"
	@echo "Run: ./$(TARGET) yamnet_core.tflite yamnet_class_map.csv wavs/miaow_16k.wav"

clean:
	rm -f $(TARGET)

# Test with sample audio
test: $(TARGET)
	@echo "Testing with sample audio..."
	LD_LIBRARY_PATH=$(TFLITE_LIB_DIR):$$LD_LIBRARY_PATH ./$(TARGET) yamnet_core.tflite yamnet_class_map.csv wavs/miaow_16k.wav

# Show build info
info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "TensorFlow: $(TENSORFLOW_ROOT)"
	@echo "TFLite lib: $(TFLITE_LIB_DIR)"
	@echo "Includes: $(INCLUDES)"

.PHONY: all clean test info