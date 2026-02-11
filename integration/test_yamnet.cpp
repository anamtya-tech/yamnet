#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <fstream>

int main() {
    printf("Loading model...\n");
    
    auto model = tflite::FlatBufferModel::BuildFromFile("yamnet.tflite");
    if (!model) {
        printf("ERROR: Failed to load yamnet.tflite\n");
        return 1;
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (!interpreter) {
        printf("ERROR: Failed to create interpreter\n");
        return 1;
    }
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Failed to allocate tensors\n");
        return 1;
    }
    
    printf("✓ Model loaded\n");
    
    // Check input
    TfLiteTensor* input = interpreter->input_tensor(0);
    int input_samples = input->bytes / sizeof(float);
    printf("✓ Expected input: %d samples (%.2f seconds at 16kHz)\n", 
           input_samples, input_samples / 16000.0);
    
    // Fill with silence
    float* input_data = input->data.f;
    for (int i = 0; i < input_samples; i++) {
        input_data[i] = 0.0f;
    }
    
    printf("Running inference...\n");
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: Inference failed\n");
        return 1;
    }
    
    // Get output
    const TfLiteTensor* output = interpreter->output_tensor(0);
    int n_frames = output->dims->data[0];
    int n_classes = output->dims->data[1];
    
    printf("✓ Output shape: [%d frames, %d classes]\n", n_frames, n_classes);
    
    // Mean aggregate
    const float* scores = output->data.f;
    std::vector<float> mean_scores(n_classes, 0.0f);
    for (int c = 0; c < n_classes; c++) {
        for (int f = 0; f < n_frames; f++) {
            mean_scores[c] += scores[f * n_classes + c];
        }
        mean_scores[c] /= n_frames;
    }
    
    int top_idx = std::max_element(mean_scores.begin(), mean_scores.end()) 
                  - mean_scores.begin();
    
    // Load class names
    std::ifstream csv("yamnet_class_map.csv");
    if (!csv.is_open()) {
        printf("WARNING: Can't load class names\n");
        printf("✓ Top class index: %d (score: %.4f)\n", top_idx, mean_scores[top_idx]);
        return 0;
    }
    
    std::string line;
    std::getline(csv, line); // skip header
    std::vector<std::string> classes;
    while (std::getline(csv, line)) {
        size_t pos = line.rfind(',');
        std::string name = line.substr(pos + 1);
        if (!name.empty() && name.front() == '"') {
            name = name.substr(1, name.size() - 2);
        }
        classes.push_back(name);
    }
    
    printf("\n=== INFERENCE SUCCESSFUL ===\n");
    printf("Predicted: %s\n", classes[top_idx].c_str());
    printf("Confidence: %.4f\n", mean_scores[top_idx]);
    
    // Top 5
    std::vector<std::pair<int, float>> ranked;
    for (int i = 0; i < n_classes; i++) {
        ranked.push_back({i, mean_scores[i]});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    
    printf("\nTop 5:\n");
    for (int i = 0; i < 5; i++) {
        printf("%d. %s (%.4f)\n", i+1, 
               classes[ranked[i].first].c_str(), 
               ranked[i].second);
    }
    
    return 0;
}
