#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>  // for std::memcpy, std::memset

class YAMNetClassifier {
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::vector<std::string> class_names;
    
    const int SAMPLE_RATE = 16000;
    const int EXPECTED_SAMPLES = 15600; // 0.975s at 16kHz. Thi is input shape yamnet expects.
    
public:
    bool LoadModel(const char* tflite_model_path) {
        model = tflite::FlatBufferModel::BuildFromFile(tflite_model_path);
        if (!model) {
            printf("Failed to load model\n");
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
        
        return true;
    }

    void PrintInputInfo() {
        TfLiteTensor* input = interpreter->input_tensor(0);
        printf("Input shape: [");
        for (int i = 0; i < input->dims->size; i++) {
            printf("%d%s", input->dims->data[i], i < input->dims->size-1 ? ", " : "");
        }
        printf("]\n");
    }

    std::string GetClassName(int idx) const {
        if (idx >= 0 && idx < class_names.size()) {
            return class_names[idx];
        }
        return "Unknown";
    }
        
    bool LoadClassNames(const char* csv_path) {
        std::ifstream file(csv_path);
        if (!file.is_open()) return false;
        
        std::string line;
        std::getline(file, line); // skip header
        
        while (std::getline(file, line)) {
            // Handle quoted CSV fields properly
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
            fields.push_back(field); // last field
            
            if (fields.size() >= 3) {
                class_names.push_back(fields[2]); // display_name is 3rd column
            }
        }
        
        return class_names.size() == 521;
    }
    
    // Normalize audio to [-1.0, 1.0]
    std::vector<float> NormalizeAudio(const int16_t* audio_data, size_t length) {
        std::vector<float> normalized(length);
        for (size_t i = 0; i < length; i++) {
            normalized[i] = audio_data[i] / 32767.0f;
        }
        return normalized;
    }
    
    // Process your ZODAS spectrum output to waveform (if needed)
    // This is a placeholder - you'd need proper IFFT
    std::vector<float> SpectrumToWaveform(float* spectrum, int frame_size) {
        // If you have time-domain audio, use that directly
        // Otherwise, implement IFFT here
        std::vector<float> waveform(EXPECTED_SAMPLES, 0.0f);
        // ... IFFT implementation ...
        return waveform;
    }
    
    bool Classify(const float* waveform, size_t length, 
              std::vector<float>& scores_out,
              int& top_class_idx,
              std::string& top_class_name) {
    
        if (length < EXPECTED_SAMPLES) {
            printf("Audio too short: %zu samples\n", length);
            return false;
        }
        
        // Process entire audio by taking many overlapping windows
        std::vector<float> accumulated_scores(521, 0.0f);
        int num_windows = 0;
        
        for (size_t offset = 0; offset + EXPECTED_SAMPLES <= length; offset += EXPECTED_SAMPLES/2) {
            TfLiteTensor* input_tensor = interpreter->input_tensor(0);
            float* input_data = input_tensor->data.f;
            
            std::memcpy(input_data, waveform + offset, EXPECTED_SAMPLES * sizeof(float));
            
            if (interpreter->Invoke() != kTfLiteOk) {
                printf("Inference failed\n");
                return false;
            }
            
            const TfLiteTensor* scores_tensor = interpreter->output_tensor(0);
            const float* scores = scores_tensor->data.f;
            
            int n_frames = scores_tensor->dims->data[0];
            int n_classes = scores_tensor->dims->data[1];
            
            // Accumulate scores
            for (int cls = 0; cls < n_classes; cls++) {
                for (int frame = 0; frame < n_frames; frame++) {
                    accumulated_scores[cls] += scores[frame * n_classes + cls];
                }
            }
            num_windows++;
        }
        
        // Average across all windows and frames
        for (int cls = 0; cls < 521; cls++) {
            accumulated_scores[cls] /= num_windows;
        }
        
        top_class_idx = std::max_element(accumulated_scores.begin(), 
                                        accumulated_scores.end()) - accumulated_scores.begin();
        if (top_class_idx >= 0 && top_class_idx < class_names.size()) {
            top_class_name = class_names[top_class_idx];
        } else {
            return false;
        }
        
        scores_out = accumulated_scores;
        return true;
    }
    
    // Integrate with your ZODAS pipeline
    bool ClassifyFromZODASSpectrum(float* spec_output, 
                                   unsigned int frame_size,
                                   std::string& result) {
        // Option 1: If you can accumulate multiple frames of spectrum
        // convert to mel spectrogram (64 bins) over time
        
        // Option 2: If you have raw audio samples, use directly
        // This is much simpler and recommended
        
        // For now, assuming you pass raw audio:
        std::vector<float> waveform(spec_output, spec_output + frame_size);
        
        std::vector<float> scores;
        int top_idx;
        std::string top_class;
        
        if (Classify(waveform.data(), waveform.size(), scores, top_idx, top_class)) {
            result = top_class;
            return true;
        }
        
        return false;
    }
};

// Usage example
int main() {
    YAMNetClassifier yamnet;
    
    // Load TFLite model (convert from TensorFlow Hub first)
    if (!yamnet.LoadModel("yamnet.tflite")) {
        return -1;
    }

    yamnet.PrintInputInfo();
    
    // Load class names
    if (!yamnet.LoadClassNames("yamnet_class_map.csv")) {
        printf("Failed to load class names\n");
        return -1;
    }
    
    // Read WAV file from wavs subfolder
    std::string wav_path = "wavs/miaow_16k.wav";
    //std::string wav_path = "wavs/speech_whistling2.wav";
    std::ifstream wav_file(wav_path, std::ios::binary);
    
    if (!wav_file.is_open()) {
        printf("Failed to open WAV file: %s\n", wav_path.c_str());
        return -1;
    }
    
    // Skip WAV header (44 bytes for standard WAV)
    wav_file.seekg(44);
    
    // Read audio samples (assuming 16-bit PCM)
    std::vector<int16_t> samples;
    int16_t sample;
    while (wav_file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        samples.push_back(sample);
    }
    wav_file.close();
    
    printf("Loaded %zu samples from %s\n", samples.size(), wav_path.c_str());
    
    // Normalize to float [-1.0, 1.0]
    std::vector<float> audio_data = yamnet.NormalizeAudio(samples.data(), samples.size());
    
    std::vector<float> scores;
    int top_class_idx;
    std::string top_class_name;
    
    if (yamnet.Classify(audio_data.data(), audio_data.size(), 
                    scores, top_class_idx, top_class_name)) {
        printf("Detected: %s (class %d, score: %.4f)\n", 
            top_class_name.c_str(), top_class_idx, scores[top_class_idx]);
        
        // Print top 5 classes
        std::vector<std::pair<int, float>> scored_classes;
        for (size_t i = 0; i < scores.size(); i++) {
            scored_classes.push_back({i, scores[i]});
        }
        std::sort(scored_classes.begin(), scored_classes.end(),
                [](auto& a, auto& b) { return a.second > b.second; });
        
        printf("\nTop 5 predictions:\n");
        for (int i = 0; i < 5 && i < scored_classes.size(); i++) {
            int idx = scored_classes[i].first;
            std::string class_name = yamnet.GetClassName(idx);
            float score = scored_classes[i].second;
            printf("%d. %s (%.4f)\n", i+1, class_name.c_str(), score);
        }
    } else {
        printf("Classification failed\n");
    }
    
    return 0;
}