/**
 * @file test_yamnet_api.cpp
 * @brief Test the YAMNetClassifier API with real audio file
 * 
 * This demonstrates the complete workflow:
 * 1. Load WAV file
 * 2. Compute FFT using Python (simulating ZODAS)
 * 3. Feed spectra to YAMNetClassifier frame-by-frame
 * 4. Get classifications
 */

#include "yamnet_classifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>

/**
 * Load pre-computed magnitude spectra from binary file
 * This simulates what ZODAS will provide
 */
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
    
    printf("✓ Loaded %d spectrum frames\n\n", num_frames);
    return true;
}

/**
 * Simple WAV to spectra using Python
 */
bool ConvertWavToSpectra(const char* wav_path, const char* spectra_path) {
    printf("Converting WAV to spectra using Python...\n");
    
    std::string cmd = "python compute_fft.py ";
    cmd += wav_path;
    cmd += " ";
    cmd += spectra_path;
    
    int result = system(cmd.c_str());
    
    if (result != 0) {
        printf("Failed to convert WAV\n");
        return false;
    }
    
    printf("✓ Conversion complete\n\n");
    return true;
}

int main(int argc, char** argv) {
    printf("==============================================\n");
    printf("YAMNet API Test\n");
    printf("==============================================\n\n");
    
    // Paths
    const char* wav_path = (argc > 1) ? argv[1] : "wavs/miaow_16k.wav";
    const char* spectra_path = "/tmp/test_spectra.bin";
    
    printf("Test audio: %s\n\n", wav_path);
    
    // Step 1: Initialize classifier
    printf("Step 1: Initializing classifier...\n");
    YAMNetClassifier classifier;
    
    if (!classifier.LoadModel("yamnet_core.tflite")) {
        printf("✗ Failed to load model\n");
        return -1;
    }
    printf("  ✓ Model loaded\n");
    
    if (!classifier.LoadClassNames("yamnet_class_map.csv")) {
        printf("✗ Failed to load class names\n");
        return -1;
    }
    printf("  ✓ Class names loaded (%d classes)\n", classifier.GetNumClasses());
    
    classifier.PrintModelInfo();
    
    // Step 2: Convert WAV to spectra (simulating ZODAS)
    printf("\nStep 2: Computing spectrum frames...\n");
    if (!ConvertWavToSpectra(wav_path, spectra_path)) {
        return -1;
    }
    
    // Step 3: Load spectra
    printf("Step 3: Loading spectra...\n");
    std::vector<std::vector<float>> all_spectra;
    if (!LoadSpectraFromFile(spectra_path, all_spectra)) {
        return -1;
    }
    
    // Step 4: Process frame-by-frame (simulating ZODAS callbacks)
    printf("Step 4: Processing frames...\n");
    printf("==============================================\n\n");
    
    int classifications_count = 0;
    
    for (size_t i = 0; i < all_spectra.size(); i++) {
        int class_id;
        std::string class_name;
        float confidence;
        
        // This is what ZODAS will call for each frame
        bool ready = classifier.AddFrame(all_spectra[i].data(), 
                                         class_id, 
                                         class_name, 
                                         confidence);
        
        if (ready) {
            classifications_count++;
            printf("[Frame %04zu] Classification #%d: %s (ID: %d, Confidence: %.2f)\n",
                   i, classifications_count, class_name.c_str(), class_id, confidence);
        }
    }
    
    printf("\n==============================================\n");
    printf("Results Summary\n");
    printf("==============================================\n");
    printf("Total frames processed: %zu\n", all_spectra.size());
    printf("Classifications made: %d\n", classifications_count);
    printf("Time between classifications: %.2f seconds\n",
           (all_spectra.size() / (float)classifications_count) * 0.01f);
    printf("\n");
    
    // Show expected results
    printf("Expected for miaow_16k.wav:\n");
    printf("  - Total frames: ~674\n");
    printf("  - Classifications: ~13\n");
    printf("  - Top classes: Animal (67), Cat (76), Speech (0)\n");
    printf("  - Most predictions should be Animal or Cat\n");
    printf("\n");
    
    // Clean up
    remove(spectra_path);
    
    printf("==============================================\n");
    printf("Test Complete!\n");
    printf("==============================================\n");
    
    return 0;
}