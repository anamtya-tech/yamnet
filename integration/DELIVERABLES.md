# YAMNet-ZODAS Integration Package

## Package Contents

This package provides everything needed to integrate YAMNet audio classification with the ZODAS audio processing pipeline.

## 📦 Deliverables

### For ZODAS Developer (Integration)

| File | Description | Required |
|------|-------------|----------|
| **QUICKSTART.md** | 3-step integration guide | ⭐ START HERE |
| **yamnet_classifier.h** | C++ header file | ✅ Include in code |
| **yamnet_classifier.cpp** | Implementation | ✅ Add to build |
| **yamnet_core.tflite** | TFLite model (10MB) | ✅ Deploy with app |
| **yamnet_class_map.csv** | 521 class names | ✅ Deploy with app |
| **zodas_integration_example.cpp** | Complete working example | 📖 Reference |
| **README.md** | Full documentation | 📖 Reference |

### For Testing & Validation

| File | Description | Purpose |
|------|-------------|---------|
| **test_pipeline.sh** | End-to-end test | Validate setup |
| **compute_fft.py** | Python FFT helper | Testing only |
| **yamnet_core_cpp.cpp** | Reference implementation | See how it works |
| **wavs/miaow_16k.wav** | Test audio file | Validation |

### For Model Development (Optional)

| File | Description | Purpose |
|------|-------------|---------|
| **export_yamnet_core.py** | Model export script | Regenerate model if needed |
| **correct_mel_computation.py** | Mel computation reference | Verify parameters |
| **analyze_predictions.py** | Prediction comparison | Debug differences |

## 🚀 Quick Integration (30 minutes)

### Step 1: Copy Files (5 min)
```bash
cp yamnet_classifier.h your_project/
cp yamnet_classifier.cpp your_project/
cp yamnet_core.tflite your_project/
cp yamnet_class_map.csv your_project/
```

### Step 2: Add to Code (10 min)
See **QUICKSTART.md** for the 3-step integration guide.

### Step 3: Build & Test (15 min)
```bash
# Build your project with added files
make

# Test with provided audio
./test_pipeline.sh wavs/miaow_16k.wav
```

## 📋 Integration Checklist

Before handoff to ZODAS developer:

- [x] Model exported and tested
- [x] C++ API created and documented
- [x] Example integration code provided
- [x] Test pipeline validated
- [x] Python FFT helper for testing
- [x] Quick start guide written
- [x] Full documentation complete

For ZODAS developer to complete:

- [ ] Copy 4 required files to project
- [ ] Add classifier to ZODAS callback
- [ ] Build with TensorFlow Lite
- [ ] Test with sample audio
- [ ] Verify 257-bin spectrum output
- [ ] Deploy to production

## 🎯 Success Criteria

Your integration is successful when:

1. ✅ Test audio (miaow_16k.wav) predicts "Animal" or "Cat"
2. ✅ Classifications appear every ~0.48 seconds
3. ✅ Processing time < 5ms per classification
4. ✅ First prediction after ~1 second
5. ✅ No "Unknown" or random classes

## 📊 Expected Behavior

### Test Audio (miaow_16k.wav)
- **Duration**: 6.73 seconds
- **Patches**: 13 classifications
- **Results**: 
  - Patch 0: Speech
  - Patches 1-12: Animal or Cat
  - Overall: Animal (class 67) or Cat (class 76)

### Timing
- **Frame input**: Every 10ms
- **Buffer time**: First 960ms (96 frames)
- **First classification**: ~1 second
- **Subsequent**: Every ~480ms (48 frames, 50% overlap)

### Performance
- **Per-frame**: < 100 μs (buffering only)
- **Per-classification**: < 5 ms (mel + inference)
- **Memory**: ~10 MB
- **CPU**: Single core sufficient

## 🔧 ZODAS Requirements

Critical parameters your ZODAS must provide:

```cpp
// Audio parameters
constexpr int SAMPLE_RATE = 16000;      // Hz
constexpr int FRAME_LENGTH = 400;       // samples (25ms)
constexpr int FRAME_STEP = 160;         // samples (10ms)
constexpr int FFT_SIZE = 512;           // points

// Output format
constexpr int SPECTRUM_BINS = 257;      // NOT 128!
float* magnitude_spectrum;              // NOT power spectrum
```

**⚠️ CRITICAL**: 257 bins required! Using 128 bins will fail.

## 📞 Support

### Documentation
1. **QUICKSTART.md** - Start here (5 min read)
2. **README.md** - Complete reference (20 min read)
3. **zodas_integration_example.cpp** - Working example

### Testing
```bash
# Validate complete pipeline
./test_pipeline.sh wavs/miaow_16k.wav

# Should output:
# - 13 patches processed
# - Overall prediction: Animal (class 67)
```

### Troubleshooting
See README.md "Troubleshooting" section for:
- Wrong predictions → Check 257 bins
- No classifications → Check timing
- Build errors → Check TFLite paths

## 📄 File Sizes

| File | Size | Type |
|------|------|------|
| yamnet_core.tflite | ~10 MB | Binary (model) |
| yamnet_class_map.csv | ~20 KB | Text |
| yamnet_classifier.h | ~12 KB | Code |
| yamnet_classifier.cpp | ~8 KB | Code |
| Total runtime | ~10 MB | - |

## 🎓 Learning Path

1. **Quick Start** (30 min)
   - Read QUICKSTART.md
   - Copy 4 files
   - Build example

2. **Integration** (2 hours)
   - Study zodas_integration_example.cpp
   - Add to your ZODAS callback
   - Build and test

3. **Validation** (1 hour)
   - Run test_pipeline.sh
   - Verify predictions
   - Check performance

4. **Production** (varies)
   - Deploy model files
   - Handle classifications
   - Monitor performance

## ✅ Ready to Deploy

This package includes:
- ✅ Production-ready C++ code
- ✅ Optimized TFLite model  
- ✅ Complete documentation
- ✅ Working examples
- ✅ Test suite
- ✅ Performance validated

The ZODAS developer has everything needed to integrate and deploy!

---

**Package Version**: 1.0  
**Date**: 2025-12-20  
**Model**: YAMNet (Google)  
**Framework**: TensorFlow Lite