# Quick Start Guide for ZODAS Integration

## For the ZODAS Developer

This guide will help you integrate YAMNet audio classification into your ZODAS pipeline in **3 simple steps**.

## Step 1: Copy Files to Your Project

Copy these files to your ZODAS project:

```
your_zodas_project/
├── yamnet_classifier.h           # Header file (include this)
├── yamnet_classifier_impl.cpp    # Implementation
├── yamnet_core.tflite           # Model file
└── yamnet_class_map.csv         # Class names
```

## Step 2: Add to Your Code

### 2.1 Include the Header

```cpp
#include "yamnet_classifier.h"
```

### 2.2 Create Classifier Instance

```cpp
YAMNetClassifier classifier;
```

### 2.3 Initialize (Once at Startup)

```cpp
bool initialize() {
    if (!classifier.LoadModel("yamnet_core.tflite")) {
        return false;
    }
    
    if (!classifier.LoadClassNames("yamnet_class_map.csv")) {
        return false;
    }
    
    return true;
}
```

### 2.4 Call from ZODAS Callback

```cpp
// Your ZODAS callback (called every 10ms with new spectrum)
void your_zodas_callback(float* spectrum_257bins) {
    int class_id;
    std::string class_name;
    float confidence;
    
    // Add frame - returns true when classification ready
    bool ready = classifier.AddFrame(spectrum_257bins, 
                                     class_id, 
                                     class_name, 
                                     confidence);
    
    if (ready) {
        // New classification available!
        printf("Detected: %s (%.2f)\n", class_name.c_str(), confidence);
        
        // Your application logic here...
    }
}
```

## Step 3: Build

### 3.1 Add to Your Makefile/CMakeLists

```makefile
SOURCES += yamnet_classifier_impl.cpp

INCLUDES += -I/path/to/tensorflow \
            -I/path/to/tensorflow/bazel-tensorflow/external/flatbuffers/include

LIBS += -L/path/to/tensorflow/bazel-bin/tensorflow/lite \
        -ltensorflowlite -ldl -lpthread
```

### 3.2 Or Build Directly

```bash
g++ -std=c++17 -O3 \
  -I/path/to/tensorflow \
  -I/path/to/tensorflow/bazel-tensorflow/external/flatbuffers/include \
  your_zodas_code.cpp \
  yamnet_classifier_impl.cpp \
  -L/path/to/tensorflow/bazel-bin/tensorflow/lite \
  -ltensorflowlite -ldl -lpthread \
  -o your_zodas_app
```

### 3.3 Run

```bash
export LD_LIBRARY_PATH=/path/to/tensorflow/bazel-bin/tensorflow/lite:$LD_LIBRARY_PATH
./your_zodas_app
```

## That's It!

You're done! The classifier will now:
- ✅ Buffer spectrum frames automatically
- ✅ Provide classifications every ~0.48 seconds
- ✅ Return class name and confidence
- ✅ Handle all mel conversion internally

## ZODAS Requirements Checklist

Make sure your ZODAS provides:

- [ ] **257 bins** (not 128!) - This is critical
- [ ] Sample rate: 16kHz
- [ ] Frame length: 400 samples (25ms)
- [ ] Frame hop: 160 samples (10ms)  
- [ ] FFT size: 512
- [ ] Window: Hann
- [ ] Output: Magnitude spectrum (not power)

## Testing

Test with the provided audio:

```bash
# Run the complete test
./test_pipeline.sh wavs/miaow_16k.wav
```

Expected: Overall prediction should be "Animal" or "Cat"

## Common Issues

**Q: All predictions are the same**
A: Check that you're providing all 257 bins, not 128

**Q: Classifications never return true**
A: Check that AddFrame() is being called every 10ms

**Q: Predictions seem wrong**  
A: Verify spectrum is magnitude (not power) and uses Hann window

## Need Help?

1. Check the full README.md for details
2. See zodas_integration_example.cpp for complete example
3. Run test_pipeline.sh to validate setup

## Files Reference

| File | Purpose |
|------|---------|
| `yamnet_classifier.h` | Include this in your code |
| `yamnet_classifier_impl.cpp` | Add to your build |
| `yamnet_core.tflite` | Model (put in runtime directory) |
| `yamnet_class_map.csv` | Class names (put in runtime directory) |
| `zodas_integration_example.cpp` | Full working example |
| `README.md` | Complete documentation |
| `test_pipeline.sh` | Test script |

## Performance

- Per-frame processing: < 100 μs
- Classification (when ready): < 5 ms  
- First result: After ~1 second (96 frames)
- Subsequent results: Every ~0.48 seconds

Happy integrating! 🎉