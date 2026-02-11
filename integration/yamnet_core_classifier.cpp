#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <complex>
#include <iostream>
#include <map>

class YAMNetCoreClassifier {
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::vector<std::string> class_names;
    
    // YAMNet parameters
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int FRAME_LENGTH = 400;  // 25ms
    static constexpr int FRAME_STEP = 160;    // 10ms
    static constexpr int FFT_SIZE = 512;
    static constexpr int NUM_MEL_BINS = 64;
    static constexpr int PATCH_FRAMES = 96;   // Number of frames per patch
    static constexpr float MEL_MIN_HZ = 125.0f;
    static constexpr float MEL_MAX_HZ = 7500.0f;
    static constexpr float LOG_OFFSET = 0.001f;
    
    std::vector<std::vector<float>> mel_filterbank;  // [64][257] matrix
    
    // Hann window
    std::vector<float> hann_window;
    
public:
    YAMNetCoreClassifier() {
        InitializeHannWindow();
        InitializeMelFilterbank();
    }
    
    bool LoadModel(const char* tflite_model_path) {
        model = tflite::FlatBufferModel::BuildFromFile(tflite_model_path);
        if (!model) {
            printf("Failed to load model from %s\n", tflite_model_path);
            return false;
        }
        
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        
        if (!interpreter) {
            printf("Failed to create interpreter\n");
            return false;
        }
        
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            printf("Failed to allocate tensors\n");
            return false;
        }
        
        printf("✓ Model loaded successfully\n");
        return true;
    }
    
    void PrintModelInfo() {
        TfLiteTensor* input = interpreter->input_tensor(0);
        TfLiteTensor* output = interpreter->output_tensor(0);
        
        printf("\nModel Information:\n");
        printf("  Input shape: [");
        for (int i = 0; i < input->dims->size; i++) {
            printf("%d%s", input->dims->data[i], i < input->dims->size-1 ? ", " : "");
        }
        printf("]\n");
        
        printf("  Output shape: [");
        for (int i = 0; i < output->dims->size; i++) {
            printf("%d%s", output->dims->data[i], i < output->dims->size-1 ? ", " : "");
        }
        printf("]\n");
        printf("  Expected input: (1, 96, 64, 1) - mel spectrogram patch\n");
        printf("  Output: (1, 521) - class probabilities\n\n");
    }
    
    bool LoadClassNames(const char* csv_path) {
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            printf("Failed to open class map: %s\n", csv_path);
            return false;
        }
        
        std::string line;
        std::getline(file, line); // skip header
        
        while (std::getline(file, line)) {
            std::vector<std::string> fields;
            std::string field;
            bool in_quotes = false;
            
            for (size_t i = 0; i < line.size(); i++) {
                char c = line[i];
                if (c == '"') {
                    in_quotes = !in_quotes;
                } else if (c == ',' && !in_quotes) {
                    fields.push_back(field);
                    field.clear();
                } else {
                    field += c;
                }
            }
            fields.push_back(field);
            
            if (fields.size() >= 3) {
                class_names.push_back(fields[2]); // display_name
            }
        }
        
        printf("✓ Loaded %zu class names\n", class_names.size());
        return class_names.size() == 521;
    }
    
private:
    void InitializeHannWindow() {
        hann_window.resize(FRAME_LENGTH);
        for (int i = 0; i < FRAME_LENGTH; i++) {
            hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (FRAME_LENGTH - 1)));
        }
    }
    
    float HzToMel(float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    }
    
    float MelToHz(float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }
    
    void InitializeMelFilterbank() {
        // Create mel filterbank: [NUM_MEL_BINS][257]
        // This converts 257 FFT bins to 64 mel bins
        
        mel_filterbank.resize(NUM_MEL_BINS);
        for (int i = 0; i < NUM_MEL_BINS; i++) {
            mel_filterbank[i].resize(257, 0.0f);
        }
        
        float mel_min = HzToMel(MEL_MIN_HZ);
        float mel_max = HzToMel(MEL_MAX_HZ);
        
        // Create mel points
        std::vector<float> mel_points(NUM_MEL_BINS + 2);
        for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
            mel_points[i] = mel_min + (mel_max - mel_min) * i / (NUM_MEL_BINS + 1);
        }
        
        // Convert mel points to FFT bin indices
        std::vector<float> bin_points(NUM_MEL_BINS + 2);
        for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
            float hz = MelToHz(mel_points[i]);
            bin_points[i] = (FFT_SIZE + 1) * hz / SAMPLE_RATE;
        }
        
        // Create triangular filters
        for (int i = 0; i < NUM_MEL_BINS; i++) {
            float left = bin_points[i];
            float center = bin_points[i + 1];
            float right = bin_points[i + 2];
            
            for (int j = 0; j < 257; j++) {
                if (j >= left && j <= center) {
                    mel_filterbank[i][j] = (j - left) / (center - left);
                } else if (j > center && j <= right) {
                    mel_filterbank[i][j] = (right - j) / (right - center);
                }
            }
        }
        
        printf("✓ Mel filterbank initialized (64 mel bins from 257 FFT bins)\n");
    }
    
    // FFT implementation - calls Python helper
    // This allows testing WAV files directly without pre-computing spectra
    void ComputeFFT(const float* frame, std::vector<float>& magnitude) {
        // NOTE: This is inefficient (calling Python per-frame)
        // Only for testing. In production, ZODAS provides spectra directly.
        
        magnitude.resize(257, 0.0f);
        
        // For now, use placeholder since calling Python per-frame is too slow
        // Use --spectra mode for accurate testing
        for (int i = 0; i < 257; i++) {
            magnitude[i] = std::abs(std::sin(i * 0.1f + frame[0])) * 0.5f;
        }
        
        static bool warning_shown = false;
        if (!warning_shown) {
            printf("\n⚠ WARNING: WAV mode uses placeholder FFT (not accurate)\n");
            printf("For accurate testing:\n");
            printf("  1. python compute_fft.py input.wav spectra.bin\n");
            printf("  2. ./yamnet_core_classifier ... --spectra spectra.bin\n\n");
            warning_shown = true;
        }
    }
    
public:
    // Process ZODAS spectrum output (257 bins) to mel spectrogram (64 bins)
    void SpectrumToMel(const std::vector<float>& magnitude_spectrum,
                      std::vector<float>& mel_out) {
        mel_out.resize(NUM_MEL_BINS, 0.0f);
        
        // Apply mel filterbank: mel[64] = filterbank[64][257] * magnitude[257]
        for (int mel_bin = 0; mel_bin < NUM_MEL_BINS; mel_bin++) {
            float sum = 0.0f;
            for (int fft_bin = 0; fft_bin < 257; fft_bin++) {
                sum += mel_filterbank[mel_bin][fft_bin] * magnitude_spectrum[fft_bin];
            }
            mel_out[mel_bin] = sum;
        }
        
        // Apply log
        for (int i = 0; i < NUM_MEL_BINS; i++) {
            mel_out[i] = std::log(mel_out[i] + LOG_OFFSET);
        }
    }
    
    // Main classification function: takes 96x257 spectrum, outputs class
    bool ClassifyFromSpectrum(const std::vector<std::vector<float>>& spectrum_frames,
                             int& top_class_idx,
                             std::string& top_class_name,
                             std::vector<float>& scores_out) {
        
        if (spectrum_frames.size() != PATCH_FRAMES) {
            printf("Expected %d frames, got %zu\n", PATCH_FRAMES, spectrum_frames.size());
            return false;
        }
        
        // Convert spectrum to mel spectrogram
        std::vector<std::vector<float>> mel_patch(PATCH_FRAMES);
        for (int frame = 0; frame < PATCH_FRAMES; frame++) {
            SpectrumToMel(spectrum_frames[frame], mel_patch[frame]);
        }
        
        // Prepare input tensor: (1, 96, 64, 1)
        TfLiteTensor* input_tensor = interpreter->input_tensor(0);
        float* input_data = input_tensor->data.f;
        
        // Fill input: [batch=1][frame=96][mel_bin=64][channel=1]
        for (int frame = 0; frame < PATCH_FRAMES; frame++) {
            for (int mel_bin = 0; mel_bin < NUM_MEL_BINS; mel_bin++) {
                int idx = frame * NUM_MEL_BINS + mel_bin;
                input_data[idx] = mel_patch[frame][mel_bin];
            }
        }
        
        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Inference failed\n");
            return false;
        }
        
        // Get output
        const TfLiteTensor* output_tensor = interpreter->output_tensor(0);
        const float* scores = output_tensor->data.f;
        
        // Copy scores
        scores_out.resize(521);
        for (int i = 0; i < 521; i++) {
            scores_out[i] = scores[i];
        }
        
        // Find top class
        top_class_idx = std::max_element(scores_out.begin(), scores_out.end()) 
                        - scores_out.begin();
        
        if (top_class_idx >= 0 && top_class_idx < class_names.size()) {
            top_class_name = class_names[top_class_idx];
        } else {
            top_class_name = "Unknown";
            return false;
        }
        
        return true;
    }
    
    // Load pre-computed magnitude spectra from binary file (Python FFT output)
    bool LoadSpectraFromFile(const char* filepath,
                            std::vector<std::vector<float>>& spectra_out) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            printf("Failed to open spectra file: %s\n", filepath);
            return false;
        }
        
        // Read dimensions
        int32_t num_frames, num_bins;
        file.read(reinterpret_cast<char*>(&num_frames), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&num_bins), sizeof(int32_t));
        
        if (num_bins != 257) {
            printf("Error: Expected 257 bins, got %d\n", num_bins);
            return false;
        }
        
        printf("Loading %d frames x %d bins from %s\n", 
               num_frames, num_bins, filepath);
        
        // Read spectra data
        spectra_out.resize(num_frames);
        for (int frame = 0; frame < num_frames; frame++) {
            spectra_out[frame].resize(257);
            file.read(reinterpret_cast<char*>(spectra_out[frame].data()), 
                     257 * sizeof(float));
        }
        
        printf("✓ Loaded %d spectrum frames\n", num_frames);
        return true;
    }
    
    // Process pre-computed spectra (simulates ZODAS output)
    bool ProcessPrecomputedSpectra(const std::vector<std::vector<float>>& all_spectra,
                                   std::vector<std::pair<int, std::string>>& predictions) {
        
        printf("\nProcessing %zu pre-computed spectrum frames...\n", all_spectra.size());
        
        // Process in patches of 96 frames with 50% overlap (hop=48)
        int patch_count = 0;
        for (size_t start = 0; start + PATCH_FRAMES <= all_spectra.size(); 
             start += 48) {
            
            std::vector<std::vector<float>> patch_spectra(
                all_spectra.begin() + start,
                all_spectra.begin() + start + PATCH_FRAMES
            );
            
            int top_class;
            std::string class_name;
            std::vector<float> scores;
            
            if (ClassifyFromSpectrum(patch_spectra, top_class, class_name, scores)) {
                predictions.push_back({top_class, class_name});
                patch_count++;
            }
        }
        
        printf("Processed %d patches\n", patch_count);
        return predictions.size() > 0;
    }
    
    // ZODAS integration: Process audio waveform frame-by-frame
    bool ProcessWaveform(const std::vector<float>& waveform,
                        std::vector<std::pair<int, std::string>>& predictions) {
        
        printf("\nProcessing waveform (%zu samples)...\n", waveform.size());
        printf("Calling Python to compute FFT...\n");
        
        // Save waveform to temporary file
        std::string temp_wav = "/tmp/temp_yamnet.wav";
        std::string temp_spec = "/tmp/temp_yamnet_spectra.bin";
        
        // Write simple WAV file
        std::ofstream wav_file(temp_wav, std::ios::binary);
        if (!wav_file.is_open()) {
            printf("Failed to create temp WAV file\n");
            return false;
        }
        
        // Write WAV header (44 bytes)
        int32_t sample_rate = 16000;
        int32_t num_samples = waveform.size();
        int32_t byte_rate = sample_rate * 2; // 16-bit mono
        int16_t block_align = 2;
        int16_t bits_per_sample = 16;
        int32_t data_size = num_samples * 2;
        int32_t file_size = 36 + data_size;
        
        wav_file.write("RIFF", 4);
        wav_file.write(reinterpret_cast<char*>(&file_size), 4);
        wav_file.write("WAVE", 4);
        wav_file.write("fmt ", 4);
        int32_t fmt_size = 16;
        wav_file.write(reinterpret_cast<char*>(&fmt_size), 4);
        int16_t audio_format = 1; // PCM
        wav_file.write(reinterpret_cast<char*>(&audio_format), 2);
        int16_t num_channels = 1;
        wav_file.write(reinterpret_cast<char*>(&num_channels), 2);
        wav_file.write(reinterpret_cast<char*>(&sample_rate), 4);
        wav_file.write(reinterpret_cast<char*>(&byte_rate), 4);
        wav_file.write(reinterpret_cast<char*>(&block_align), 2);
        wav_file.write(reinterpret_cast<char*>(&bits_per_sample), 2);
        wav_file.write("data", 4);
        wav_file.write(reinterpret_cast<char*>(&data_size), 4);
        
        // Write samples
        for (float sample : waveform) {
            int16_t sample_int = static_cast<int16_t>(sample * 32767.0f);
            wav_file.write(reinterpret_cast<char*>(&sample_int), 2);
        }
        wav_file.close();
        
        // Call Python to compute FFT
        std::string python_cmd = "python compute_fft.py " + temp_wav + " " + temp_spec;
        int result = system(python_cmd.c_str());
        
        if (result != 0) {
            printf("Python FFT computation failed\n");
            return false;
        }
        
        // Load computed spectra
        std::vector<std::vector<float>> all_spectra;
        if (!LoadSpectraFromFile(temp_spec.c_str(), all_spectra)) {
            return false;
        }
        
        // Clean up temp files
        remove(temp_wav.c_str());
        remove(temp_spec.c_str());
        
        // Process spectra
        return ProcessPrecomputedSpectra(all_spectra, predictions);
    }
    
    std::string GetClassName(int idx) const {
        if (idx >= 0 && idx < class_names.size()) {
            return class_names[idx];
        }
        return "Unknown";
    }
};

// Test with WAV file
std::vector<float> LoadWAV(const std::string& filepath, int& sample_rate) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        printf("Failed to open: %s\n", filepath.c_str());
        return {};
    }
    
    // Read WAV header
    char header[44];
    file.read(header, 44);
    
    // Extract sample rate from header (bytes 24-27)
    sample_rate = *reinterpret_cast<int*>(&header[24]);
    
    // Read samples
    std::vector<int16_t> samples;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        samples.push_back(sample);
    }
    
    // Normalize to [-1, 1]
    std::vector<float> normalized(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        normalized[i] = samples[i] / 32767.0f;
    }
    
    printf("✓ Loaded %zu samples from %s (%.2fs at %dHz)\n", 
           samples.size(), filepath.c_str(), 
           samples.size() / (float)sample_rate, sample_rate);
    
    return normalized;
}

int main(int argc, char** argv) {
    printf("==============================================\n");
    printf("YAMNet Core Classifier (C++)\n");
    printf("==============================================\n\n");
    
    if (argc < 4) {
        printf("Usage:\n");
        printf("  Mode 1 (WAV file - placeholder FFT):\n");
        printf("    %s <model.tflite> <class_map.csv> <audio.wav>\n\n", argv[0]);
        printf("  Mode 2 (Pre-computed spectra - accurate):\n");
        printf("    %s <model.tflite> <class_map.csv> --spectra <spectra.bin>\n\n", argv[0]);
        printf("To generate spectra file:\n");
        printf("    python compute_fft.py <audio.wav> <spectra.bin>\n\n");
        return -1;
    }
    
    // Paths
    const char* model_path = argv[1];
    const char* class_map_path = argv[2];
    const char* input_path = argv[3];
    
    // Check mode
    bool use_precomputed = (argc > 4 && std::string(argv[3]) == "--spectra");
    if (use_precomputed) {
        input_path = argv[4];
    }
    
    // Initialize classifier
    YAMNetCoreClassifier classifier;
    
    if (!classifier.LoadModel(model_path)) {
        return -1;
    }
    
    classifier.PrintModelInfo();
    
    if (!classifier.LoadClassNames(class_map_path)) {
        return -1;
    }
    
    // Process based on mode
    std::vector<std::pair<int, std::string>> predictions;
    
    printf("\n==============================================\n");
    printf("Processing...\n");
    printf("==============================================\n");
    
    if (use_precomputed) {
        // Mode 2: Load pre-computed spectra from Python FFT
        printf("Mode: Pre-computed spectra (accurate)\n\n");
        
        std::vector<std::vector<float>> spectra;
        if (!classifier.LoadSpectraFromFile(input_path, spectra)) {
            return -1;
        }
        
        if (!classifier.ProcessPrecomputedSpectra(spectra, predictions)) {
            printf("Processing failed\n");
            return -1;
        }
        
    } else {
        // Mode 1: Load WAV and use placeholder FFT
        printf("Mode: WAV file (placeholder FFT - for testing only)\n");
        printf("For accurate results, use pre-computed spectra mode\n\n");
        
        int sample_rate;
        std::vector<float> waveform = LoadWAV(input_path, sample_rate);
        
        if (waveform.empty()) {
            return -1;
        }
        
        if (sample_rate != 16000) {
            printf("WARNING: Sample rate is %dHz, expected 16000Hz\n", sample_rate);
        }
        
        if (!classifier.ProcessWaveform(waveform, predictions)) {
            printf("Processing failed\n");
            return -1;
        }
    }
    
    // Display results
    printf("\n==============================================\n");
    printf("Results (%zu patches)\n", predictions.size());
    printf("==============================================\n");
    
    for (size_t i = 0; i < predictions.size(); i++) {
        printf("Patch %2zu: Class %3d - %s\n", 
               i, predictions[i].first, predictions[i].second.c_str());
    }
    
    // Find most common prediction
    std::map<int, int> class_counts;
    for (const auto& pred : predictions) {
        class_counts[pred.first]++;
    }
    
    int most_common = std::max_element(
        class_counts.begin(), class_counts.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    )->first;
    
    printf("\n==============================================\n");
    printf("Overall prediction: %s (class %d)\n", 
           classifier.GetClassName(most_common).c_str(), most_common);
    printf("==============================================\n");
    
    return 0;
}