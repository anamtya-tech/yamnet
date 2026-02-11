# Building tflite

## install repo
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

## inastall bazel
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
npm install -g @bazel/bazelisk

## build

Use XNNPACK Delegate: This is essential for ARM CPUs. It uses ARM Neon instructions to speed up inference significantly. Add this flag to your Bazel command:
--define tflite_with_xnnpack=true


bazel build -c opt --define tflite_with_xnnpack=true //tensorflow/lite:libtensorflowlite.so


### use this for cross compiling for Raspberry PI 64 bit
bazel build -c opt --config=elinux_aarch64 --define tflite_with_xnnpack=true //tensorflow/lite:libtensorflowlite.so


OR for old 32 bit:

bazel build --config=elinux_armhf -c opt //tensorflow/lite:libtensorflowlite.so

Note: If you are using a quantized model, you should also include --define tflite_with_xnnpack_qs8=true to enable XNNPACK for signed 8-bit integer operations. 

Overclocking: Increasing the Pi 4 CPU to 1.95 GHz has been shown to boost frame rates for similar models (like MobileNet) by over 20%.

Quantization: For YAMNet, using a quantized (.tflite) model is highly recommended for Pi deployment. It reduces memory usage and allows the CPU to process integer operations faster than floats.


## Deployment
Once the build is complete:
Locate the library in bazel-bin/tensorflow/lite/libtensorflowlite.so.
Copy this file to your Raspberry Pi.
Link it to your C++ project on the Pi using a standard C++ compiler (like g++). Ensure you have the FlatBuffers and Abseil headers available, as TFLite depends on them. 

LD_LIBRARY_PATH=/home/azureuser/tensorflow/bazel-bin/tensorflow/lite:$LD_LIBRARY_PATH ./yamnet_classifier