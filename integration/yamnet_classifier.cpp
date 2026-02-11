/**
 * @file yamnet_classifier_impl.cpp
 * @brief Implementation of YAMNetClassifier using pImpl pattern
 * 
 * This keeps TensorFlow Lite headers out of the public API.
 */

#include "yamnet_classifier.h"
#include <tensorflow/lite/interpreter.h> 
#include <tensorflow/lite/model.h> 
#include <tensorflow/lite/kernels/register.h> 
#include <tensorflow/lite/op_resolver.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

// Implementation class (hidden from header)
class YAMNetClassifier::Impl {
public:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::vector<std::string> class_names;
    std::vector<std::vector<float>> mel_filterbank;
    std::vector<float> hann_window;
    
    // Frame buffer for accumulating spectra
    std::vector<std::vector<float>> frame_buffer;
    int frames_since_last_classification;
    
    Impl() : frames_since_last_classification(0) {
        InitializeHannWindow();
        InitializeMelFilterbank();
    }
    
    void InitializeHannWindow() {
        hann_window.resize(YAMNet::Params::FRAME_LENGTH);
        for (int i = 0; i < YAMNet::Params::FRAME_LENGTH; i++) {
            hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / 
                                    (YAMNet::Params::FRAME_LENGTH - 1)));
        }
    }
    
    float HzToMel(float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    }
    
    float MelToHz(float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }
    
    void InitializeMelFilterbank() {
        mel_filterbank.resize(YAMNet::Params::MEL_BINS);
        for (int i = 0; i < YAMNet::Params::MEL_BINS; i++) {
            mel_filterbank[i].resize(YAMNet::Params::SPECTRUM_BINS, 0.0f);
        }
        
        float mel_min = HzToMel(YAMNet::Params::MEL_MIN_HZ);
        float mel_max = HzToMel(YAMNet::Params::MEL_MAX_HZ);
        
        std::vector<float> mel_points(YAMNet::Params::MEL_BINS + 2);
        for (int i = 0; i < YAMNet::Params::MEL_BINS + 2; i++) {
            mel_points[i] = mel_min + (mel_max - mel_min) * i / 
                           (YAMNet::Params::MEL_BINS + 1);
        }
        
        std::vector<float> bin_points(YAMNet::Params::MEL_BINS + 2);
        for (int i = 0; i < YAMNet::Params::MEL_BINS + 2; i++) {
            float hz = MelToHz(mel_points[i]);
            bin_points[i] = (YAMNet::Params::FFT_SIZE + 1) * hz / 
                           YAMNet::Params::SAMPLE_RATE;
        }
        
        for (int i = 0; i < YAMNet::Params::MEL_BINS; i++) {
            float left = bin_points[i];
            float center = bin_points[i + 1];
            float right = bin_points[i + 2];
            
            for (int j = 0; j < YAMNet::Params::SPECTRUM_BINS; j++) {
                if (j >= left && j <= center) {
                    mel_filterbank[i][j] = (j - left) / (center - left);
                } else if (j > center && j <= right) {
                    mel_filterbank[i][j] = (right - j) / (right - center);
                }
            }
        }
    }
    
    void SpectrumToMel(const std::vector<float>& magnitude_spectrum,
                      std::vector<float>& mel_out) {
        mel_out.resize(YAMNet::Params::MEL_BINS, 0.0f);
        
        for (int mel_bin = 0; mel_bin < YAMNet::Params::MEL_BINS; mel_bin++) {
            float sum = 0.0f;
            for (int fft_bin = 0; fft_bin < YAMNet::Params::SPECTRUM_BINS; fft_bin++) {
                sum += mel_filterbank[mel_bin][fft_bin] * magnitude_spectrum[fft_bin];
            }
            mel_out[mel_bin] = sum;
        }
        
        for (int i = 0; i < YAMNet::Params::MEL_BINS; i++) {
            mel_out[i] = std::log(mel_out[i] + YAMNet::Params::LOG_OFFSET);
        }
    }
    
    bool ClassifyFromBuffer(int& class_id_out,
                           std::string& class_name_out,
                           float& confidence_out) {
        
        if (frame_buffer.size() < YAMNet::Params::PATCH_FRAMES) {
            return false;
        }
        
        // Convert last 96 frames to mel
        std::vector<std::vector<float>> mel_patch(YAMNet::Params::PATCH_FRAMES);
        size_t start_idx = frame_buffer.size() - YAMNet::Params::PATCH_FRAMES;
        
        for (int i = 0; i < YAMNet::Params::PATCH_FRAMES; i++) {
            SpectrumToMel(frame_buffer[start_idx + i], mel_patch[i]);
        }
        
        // Fill input tensor
        TfLiteTensor* input_tensor = interpreter->input_tensor(0);
        float* input_data = input_tensor->data.f;
        
        for (int frame = 0; frame < YAMNet::Params::PATCH_FRAMES; frame++) {
            for (int mel_bin = 0; mel_bin < YAMNet::Params::MEL_BINS; mel_bin++) {
                int idx = frame * YAMNet::Params::MEL_BINS + mel_bin;
                input_data[idx] = mel_patch[frame][mel_bin];
            }
        }
        
        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            return false;
        }
        
        // Get output
        const TfLiteTensor* output_tensor = interpreter->output_tensor(0);
        const float* scores = output_tensor->data.f;
        
        // Find top class
        int top_idx = 0;
        float top_score = scores[0];
        
        for (int i = 1; i < YAMNet::Params::NUM_CLASSES; i++) {
            if (scores[i] > top_score) {
                top_score = scores[i];
                top_idx = i;
            }
        }
        
        class_id_out = top_idx;
        class_name_out = (top_idx < class_names.size()) ? 
                         class_names[top_idx] : "Unknown";
        confidence_out = top_score;
        
        return true;
    }
};

// ============================================================================
// Public API Implementation
// ============================================================================

YAMNetClassifier::YAMNetClassifier() 
    : pImpl(std::make_unique<Impl>()) {
}

YAMNetClassifier::~YAMNetClassifier() = default;

bool YAMNetClassifier::LoadModel(const char* tflite_model_path) {
    pImpl->model = tflite::FlatBufferModel::BuildFromFile(tflite_model_path);
    if (!pImpl->model) {
        std::cerr << "Failed to load model from " << tflite_model_path << std::endl;
        return false;
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*pImpl->model, resolver)(&pImpl->interpreter);
    
    if (!pImpl->interpreter) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return false;
    }
    
    if (pImpl->interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return false;
    }
    
    return true;
}

bool YAMNetClassifier::LoadClassNames(const char* csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open class map: " << csv_path << std::endl;
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
            pImpl->class_names.push_back(fields[2]); // display_name
        }
    }
    
    return pImpl->class_names.size() == YAMNet::Params::NUM_CLASSES;
}

bool YAMNetClassifier::AddFrame(const float* magnitude_spectrum_257bins,
                                int& class_id_out,
                                std::string& class_name_out,
                                float& confidence_out) {
    
    // Add frame to buffer
    std::vector<float> frame(magnitude_spectrum_257bins, 
                            magnitude_spectrum_257bins + YAMNet::Params::SPECTRUM_BINS);
    pImpl->frame_buffer.push_back(frame);
    
    // Keep only necessary frames (96 for current + 48 for overlap)
    if (pImpl->frame_buffer.size() > YAMNet::Params::PATCH_FRAMES + YAMNet::Params::PATCH_HOP) {
        pImpl->frame_buffer.erase(pImpl->frame_buffer.begin(), 
                                 pImpl->frame_buffer.begin() + YAMNet::Params::PATCH_HOP);
    }
    
    pImpl->frames_since_last_classification++;
    
    // First classification when we have 96 frames
    if (pImpl->frame_buffer.size() == YAMNet::Params::PATCH_FRAMES) {
        pImpl->frames_since_last_classification = 0;
        return pImpl->ClassifyFromBuffer(class_id_out, class_name_out, confidence_out);
    }
    
    // Subsequent classifications every 48 frames (50% overlap)
    if (pImpl->frame_buffer.size() > YAMNet::Params::PATCH_FRAMES &&
        pImpl->frames_since_last_classification >= YAMNet::Params::PATCH_HOP) {
        pImpl->frames_since_last_classification = 0;
        return pImpl->ClassifyFromBuffer(class_id_out, class_name_out, confidence_out);
    }
    
    return false;
}

void YAMNetClassifier::Reset() {
    pImpl->frame_buffer.clear();
    pImpl->frames_since_last_classification = 0;
}

std::string YAMNetClassifier::GetClassName(int class_id) const {
    if (class_id >= 0 && class_id < static_cast<int>(pImpl->class_names.size())) {
        return pImpl->class_names[class_id];
    }
    return "Unknown";
}

int YAMNetClassifier::GetNumClasses() const {
    return pImpl->class_names.size();
}

bool YAMNetClassifier::IsReady() const {
    return pImpl->interpreter != nullptr && 
           pImpl->class_names.size() == YAMNet::Params::NUM_CLASSES;
}

void YAMNetClassifier::PrintModelInfo() const {
    if (!pImpl->interpreter) {
        std::cout << "Model not loaded" << std::endl;
        return;
    }
    
    TfLiteTensor* input = pImpl->interpreter->input_tensor(0);
    TfLiteTensor* output = pImpl->interpreter->output_tensor(0);
    
    std::cout << "Model Information:" << std::endl;
    std::cout << "  Input shape: [";
    for (int i = 0; i < input->dims->size; i++) {
        std::cout << input->dims->data[i];
        if (i < input->dims->size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Output shape: [";
    for (int i = 0; i < output->dims->size; i++) {
        std::cout << output->dims->data[i];
        if (i < output->dims->size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Classes loaded: " << pImpl->class_names.size() << std::endl;
}