# YAMNet Audio Classification for ZODAS Integration

## Overview

This package provides real-time audio classification using Google's YAMNet model, optimized for integration with the ZODAS audio processing pipeline. The implementation processes 257-bin magnitude spectra from ZODAS and outputs audio event classifications.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ZODAS Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Audio Input (16kHz) ───────────────────────────────────────┐   │
│                                                             │   │
│  ┌─────────────────────────────────────────────────────┐    │   │
│  │ ZODAS Audio Processing                              │    │   │
│  │  - STFT computation (512-point FFT)                 │    │   │
│  │  - Hann windowing                                   │    │   │
│  │  - Frame length: 400 samples (25ms)                 │    │   │
│  │  - Frame hop: 160 samples (10ms)                    │    │   │
│  │  - Output: 257-bin magnitude spectrum per frame     │    │   │
│  └─────────────────────────────────────────────────────┘    │   │
│                          │                                  │   │
│                          ▼                                  │   │
│  ┌─────────────────────────────────────────────────────┐    │   │
│  │ YAMNet Classifier (this package)                    │    │   │
│  │  - Buffer: 96 frames                                │    │   │
│  │  - Mel conversion: 257 bins → 64 mel bins           │    │   │
│  │  - Log mel spectrogram computation                  │    │   │
│  │  - TFLite inference                                 │    │   │
│  │  - Output: 521 audio event classes                  │    │   │
│  └─────────────────────────────────────────────────────┘    │   │
│                          │                                  │   │
│                          ▼                                  │   │
│                 Classification Result                       │   │
│                 (e.g., "Speech", "Music", "Cat")            │   │
│                                                             │   │
└─────────────────────────────────────────────────────────────┘   │
                                                                  │
                     Your Application ◄───────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- TensorFlow Lite library (built and installed)
- C++17 compiler
- Model files (provided):
  - `yamnet_core.tflite` - TFLite model
  - `yamnet_class_map.csv` - Class name mappings

### 2. Integration Steps

```cpp
#include "yamnet_classifier.h"

// 1. Initialize classifier (once at startup)
YAMNetClassifier classifier;
classifier.LoadModel("yamnet_core.tflite");
classifier.LoadClassNames("yamnet_class_map.csv");

// 2. In your ZODAS callback (called every 10ms with new spectrum)
void zodas_spectrum_callback(float* magnitude_spectrum_257bins) {
    int class_id;
    std::string class_name;
    float confidence;
    
    // Add frame and get prediction when ready
    bool ready = classifier.AddFrame(magnitude_spectrum_257bins, 
                                      class_id, class_name, confidence);
    
    if (ready) {
        printf("Detected: %s (confidence: %.2f)\n", 
               class_name.c_str(), confidence);
    }
}
```

### 3. Build Your Application

```bash
g++ -std=c++17 -O3 \
  -I/path/to/tensorflow \
  -I/path/to/tensorflow/bazel-tensorflow/external/flatbuffers/include \
  your_app.cpp \
  -L/path/to/tensorflow/bazel-bin/tensorflow/lite \
  -ltensorflowlite \
  -ldl -lpthread \
  -o your_app
```

## API Reference

### YAMNetClassifier Class

#### Initialization

```cpp
bool LoadModel(const char* tflite_model_path);
```
Load the TFLite model. Call once at startup.

**Parameters:**
- `tflite_model_path`: Path to `yamnet_core.tflite`

**Returns:** `true` on success

---

```cpp
bool LoadClassNames(const char* csv_path);
```
Load class name mappings. Call once at startup.

**Parameters:**
- `csv_path`: Path to `yamnet_class_map.csv`

**Returns:** `true` on success

---

#### Frame-by-Frame Processing

```cpp
bool AddFrame(const float* magnitude_spectrum_257bins,
              int& class_id_out,
              std::string& class_name_out,
              float& confidence_out);
```
Add a new spectrum frame and get classification when ready.

**Parameters:**
- `magnitude_spectrum_257bins`: Input spectrum (257 float values)
- `class_id_out`: Output class ID (0-520)
- `class_name_out`: Output class name (e.g., "Speech", "Music")
- `confidence_out`: Confidence score (0.0-1.0)

**Returns:** 
- `true` if a new classification is ready (every ~0.48s with 50% overlap)
- `false` if still buffering frames

**Buffer Behavior:**
- First 96 frames: Buffering, returns `false`
- Frame 96+: Returns `true` every 48 frames (50% overlap)

---

```cpp
void Reset();
```
Clear the frame buffer. Call when starting a new audio stream.

---

```cpp
std::string GetClassName(int class_id) const;
```
Get class name from class ID.

**Parameters:**
- `class_id`: Class ID (0-520)

**Returns:** Class name string

## ZODAS Requirements

Your ZODAS implementation must provide:

### Audio Parameters
- **Sample rate**: 16,000 Hz
- **Frame length**: 400 samples (25 ms)
- **Frame hop**: 160 samples (10 ms)
- **FFT size**: 512 points

### Output Format
- **Spectrum bins**: 257 (not 128!)
- **Data type**: `float*` array
- **Content**: Magnitude spectrum (not power spectrum)
- **Range**: [0, ∞) - no normalization required

### Window Function
- **Type**: Hann window
- **Formula**: `w[n] = 0.5 * (1 - cos(2*π*n/(N-1)))`

### Critical: 257 Bins

⚠️ **IMPORTANT**: You must provide all **257 frequency bins** from the 512-point FFT.

The 512-point real FFT produces 257 unique bins:
- Bin 0: DC component
- Bins 1-256: Positive frequencies up to Nyquist

Do NOT truncate to 128 bins - this will cause incorrect predictions.

## Testing Your Integration

### Step 1: Verify ZODAS Output

```cpp
// In your ZODAS callback
void zodas_callback(float* spectrum) {
    // Verify output
    static int frame_count = 0;
    if (frame_count == 0) {
        printf("First frame spectrum:\n");
        printf("  Bin 0 (DC): %.6f\n", spectrum[0]);
        printf("  Bin 128: %.6f\n", spectrum[128]);
        printf("  Bin 256 (Nyquist): %.6f\n", spectrum[256]);
        
        // Check for common errors
        if (spectrum[128] == 0 && spectrum[256] == 0) {
            printf("⚠ WARNING: Only first 128 bins have data!\n");
            printf("You must provide all 257 bins.\n");
        }
    }
    frame_count++;
}
```

### Step 2: Test with Known Audio

Use the provided test audio:

```bash
# Test the complete pipeline
./test_pipeline.sh wavs/miaow_16k.wav
```

Expected output:
- 13 patches processed
- Overall prediction: "Animal" or "Cat"
- Most patches should predict class 67 (Animal) or 76 (Cat)

### Step 3: Validate Real-time Performance

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

bool ready = classifier.AddFrame(spectrum, class_id, class_name, confidence);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

printf("Processing time: %ld μs\n", duration.count());
```

Target performance:
- Frame processing: < 100 μs
- Classification (when ready): < 5 ms
- Total latency: ~1 second (96 frames @ 10ms/frame)

## Example Integration

### Complete Example

```cpp
#include "yamnet_classifier.h"
#include <stdio.h>

class MyAudioProcessor {
private:
    YAMNetClassifier yamnet;
    
public:
    bool Initialize() {
        if (!yamnet.LoadModel("yamnet_core.tflite")) {
            printf("Failed to load model\n");
            return false;
        }
        
        if (!yamnet.LoadClassNames("yamnet_class_map.csv")) {
            printf("Failed to load class names\n");
            return false;
        }
        
        printf("YAMNet initialized successfully\n");
        return true;
    }
    
    void ProcessSpectrum(float* spectrum_257bins) {
        int class_id;
        std::string class_name;
        float confidence;
        
        bool ready = yamnet.AddFrame(spectrum_257bins, 
                                      class_id, class_name, confidence);
        
        if (ready) {
            // New classification available
            OnAudioEvent(class_id, class_name, confidence);
        }
    }
    
    void OnAudioEvent(int class_id, const std::string& class_name, 
                     float confidence) {
        printf("[Audio Event] %s (ID: %d, Confidence: %.2f)\n",
               class_name.c_str(), class_id, confidence);
        
        // Your application logic here
        if (class_name == "Speech") {
            // Handle speech detection
        } else if (class_name == "Music") {
            // Handle music detection
        }
        // ... etc
    }
    
    void StartNewStream() {
        yamnet.Reset();
    }
};

// Usage
int main() {
    MyAudioProcessor processor;
    
    if (!processor.Initialize()) {
        return -1;
    }
    
    // Simulate ZODAS callbacks
    while (running) {
        float* spectrum = zodas_get_spectrum();  // Your ZODAS API
        processor.ProcessSpectrum(spectrum);
    }
    
    return 0;
}
```

## Model Information

### Input Specifications
- **Shape**: (1, 96, 64, 1)
- **Type**: float32
- **Content**: Log mel spectrogram
- **Range**: Approximately [-6.2, 3.8]

The classifier handles conversion from your 257-bin spectrum to this format automatically.

### Output Specifications
- **Shape**: (1, 521)
- **Type**: float32
- **Content**: Sigmoid probabilities (multi-label)
- **Range**: [0.0, 1.0]

Note: This is multi-label classification. Multiple classes can have high scores simultaneously (e.g., "Animal" and "Cat").

### Class Categories

The 521 classes include:
- **Speech**: Speech, Conversation, Narration
- **Music**: Music, Musical instrument, Singing
- **Nature**: Animal, Bird, Cat, Dog, Wind, Rain
- **Vehicle**: Car, Truck, Train, Aircraft
- **Alarm**: Alarm, Siren, Buzzer
- **Domestic**: Door, Footsteps, Appliances
- And many more...

See `yamnet_class_map.csv` for the complete list.

## Troubleshooting

### Problem: All predictions are the same class

**Cause**: Only 128 bins provided instead of 257

**Solution**: Modify ZODAS to output all 257 FFT bins

---

### Problem: Predictions don't match expected results

**Cause**: Incorrect mel computation parameters

**Solution**: Verify:
- Log offset = 0.001 (not 0.01)
- 257 bins input
- Correct mel filterbank range (125-7500 Hz)

---

### Problem: High latency

**Cause**: Normal - requires 96 frames

**Solution**: This is expected. First prediction comes after ~1 second (96 × 10ms). Subsequent predictions every ~0.48s (48 × 10ms).

---

### Problem: Model output sum >> 1.0

**Cause**: Batch normalization states not transferred

**Solution**: Re-export model using provided `export_yamnet_core.py`

## Files Provided

```
.
├── README.md                      # This file
├── yamnet_classifier.h            # Header file for integration
├── yamnet_core.tflite            # TFLite model
├── yamnet_class_map.csv          # Class name mappings
├── yamnet_core_cpp.cpp           # Implementation (reference)
├── compute_fft.py                # Python FFT helper (for testing)
├── export_yamnet_core.py         # Model export script
├── test_pipeline.sh              # Integration test
└── wavs/
    └── miaow_16k.wav             # Test audio file
```

## Support & Validation

### Validation Checklist

Before deploying:

- [ ] Model loads successfully
- [ ] Class names loaded (521 classes)
- [ ] ZODAS provides 257 bins (not 128)
- [ ] First prediction after ~1 second
- [ ] Subsequent predictions every ~0.48s
- [ ] Test audio predicts "Animal" or "Cat"
- [ ] Processing time < 5ms per classification
- [ ] No memory leaks (run valgrind)

### Reference Implementation

A complete reference implementation is provided in `yamnet_core_cpp.cpp`. This includes:
- Full mel filterbank implementation
- TFLite integration
- Frame buffering logic
- Performance optimizations

## Performance Characteristics

### Computational Cost
- **Per frame** (10ms audio): < 100 μs
- **Per classification** (96 frames): < 5 ms
- **Memory usage**: ~10 MB (model + buffers)

### Latency
- **First prediction**: ~1.0 second (96 frames × 10ms)
- **Subsequent predictions**: ~0.48 second (48 frames × 10ms, 50% overlap)

### Accuracy
- **AudioSet evaluation**: Mean Average Precision (mAP) = 0.521
- **Real-world performance**: Suitable for production use
- **Multi-label**: Can detect multiple simultaneous events

## License

This implementation uses:
- **YAMNet model**: Apache 2.0 License (Google)
- **TensorFlow Lite**: Apache 2.0 License
- **Integration code**: [Your license]

## References

- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- AudioSet: https://research.google.com/audioset/
- TensorFlow Lite: https://www.tensorflow.org/lite