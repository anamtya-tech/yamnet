/**
 * @file zodas_integration_example.cpp
 * @brief Example integration of YAMNet with ZODAS audio pipeline
 * 
 * This example shows how to integrate the YAMNet classifier with your
 * ZODAS audio processing pipeline for real-time audio event detection.
 */

#include "yamnet_classifier.h"
#include <stdio.h>
#include <vector>
#include <cmath>
#include <unistd.h>

/**
 * Example ZODAS integration class
 */
class ZODASAudioProcessor {
private:
    YAMNetClassifier yamnet;
    
    // Statistics
    int total_frames_processed;
    int classifications_made;
    
public:
    ZODASAudioProcessor() 
        : total_frames_processed(0), classifications_made(0) {
    }
    
    /**
     * Initialize the audio classifier
     * Call this once at startup
     */
    bool Initialize() {
        printf("Initializing YAMNet classifier...\n");
        
        // Load model
        if (!yamnet.LoadModel("yamnet_core.tflite")) {
            printf("ERROR: Failed to load model\n");
            return false;
        }
        printf("  ✓ Model loaded\n");
        
        // Load class names
        if (!yamnet.LoadClassNames("yamnet_class_map.csv")) {
            printf("ERROR: Failed to load class names\n");
            return false;
        }
        printf("  ✓ Class names loaded (%d classes)\n", yamnet.GetNumClasses());
        
        // Print model info
        yamnet.PrintModelInfo();
        
        printf("YAMNet initialization complete\n\n");
        return true;
    }
    
    /**
     * Process a spectrum frame from ZODAS
     * 
     * This function should be called from your ZODAS callback
     * every time a new spectrum frame is available (every 10ms).
     * 
     * @param spectrum_257bins The 257-bin magnitude spectrum from ZODAS
     */
    void ProcessFrame(const float* spectrum_257bins) {
        total_frames_processed++;
        
        int class_id;
        std::string class_name;
        float confidence;
        
        // Add frame and check if classification is ready
        bool classification_ready = yamnet.AddFrame(spectrum_257bins, 
                                                    class_id, 
                                                    class_name, 
                                                    confidence);
        
        if (classification_ready) {
            classifications_made++;
            OnAudioEventDetected(class_id, class_name, confidence);
        }
    }
    
    /**
     * Callback for when an audio event is detected
     * 
     * Override this function to handle detected audio events in your
     * application (e.g., trigger actions, log events, update UI, etc.)
     */
    virtual void OnAudioEventDetected(int class_id, 
                                     const std::string& class_name,
                                     float confidence) {
        printf("[%05d frames] Audio Event: %s (ID: %d, Confidence: %.2f)\n",
               total_frames_processed, class_name.c_str(), class_id, confidence);
        
        // Example: Handle specific events
        if (class_name == "Speech" && confidence > 0.3) {
            HandleSpeechDetected(confidence);
        } else if (class_name == "Music" && confidence > 0.3) {
            HandleMusicDetected(confidence);
        } else if (class_id == YAMNet::ClassID::ALARM) {
            HandleAlarmDetected(confidence);
        }
    }
    
    /**
     * Example event handlers - implement these based on your needs
     */
    void HandleSpeechDetected(float confidence) {
        // Your application logic
        // e.g., start voice recognition, mute audio output, etc.
    }
    
    void HandleMusicDetected(float confidence) {
        // Your application logic
        // e.g., adjust equalizer, enable music mode, etc.
    }
    
    void HandleAlarmDetected(float confidence) {
        // Your application logic
        // e.g., alert user, trigger recording, etc.
    }
    
    /**
     * Reset the classifier
     * Call this when starting a new audio stream
     */
    void Reset() {
        yamnet.Reset();
        total_frames_processed = 0;
        classifications_made = 0;
        printf("Classifier reset\n");
    }
    
    /**
     * Get processing statistics
     */
    void PrintStats() const {
        printf("\nProcessing Statistics:\n");
        printf("  Total frames: %d (%.2f seconds)\n", 
               total_frames_processed,
               total_frames_processed * 0.01f);
        printf("  Classifications: %d\n", classifications_made);
        printf("  Avg time between classifications: %.2f ms\n",
               total_frames_processed > 0 ? 
               (total_frames_processed * 10.0f) / classifications_made : 0);
    }
};


// ============================================================================
// Example 1: Simulated ZODAS callback
// ============================================================================

ZODASAudioProcessor* g_processor = nullptr;

/**
 * This is what your ZODAS callback might look like
 * This gets called every 10ms with a new spectrum
 */
void zodas_spectrum_callback(float* magnitude_spectrum_257bins) {
    if (g_processor) {
        g_processor->ProcessFrame(magnitude_spectrum_257bins);
    }
}


// ============================================================================
// Example 2: Main program
// ============================================================================

int main(int argc, char** argv) {
    printf("==============================================\n");
    printf("YAMNet + ZODAS Integration Example\n");
    printf("==============================================\n\n");
    
    // Create and initialize processor
    ZODASAudioProcessor processor;
    g_processor = &processor;
    
    if (!processor.Initialize()) {
        return -1;
    }
    
    printf("==============================================\n");
    printf("Processing Audio\n");
    printf("==============================================\n");
    printf("Waiting for ZODAS spectrum data...\n\n");
    
    // TODO: Replace this simulation with your actual ZODAS integration
    // In your real code, ZODAS will call zodas_spectrum_callback()
    // automatically whenever a new spectrum is available.
    
    // SIMULATION: Generate some test frames
    // In production, remove this and let ZODAS call zodas_spectrum_callback()
    for (int i = 0; i < 200; i++) {  // Simulate 2 seconds of audio
        float test_spectrum[257];
        
        // Generate fake spectrum (your ZODAS will provide real data)
        for (int j = 0; j < 257; j++) {
            test_spectrum[j] = std::abs(std::sin(i * 0.1f + j * 0.01f));
        }
        
        // This is what ZODAS will call
        zodas_spectrum_callback(test_spectrum);
        
        // Simulate 10ms delay between frames
        // (not needed in production - ZODAS timing handles this)
        usleep(10000);  // 10ms
    }
    
    printf("\n");
    processor.PrintStats();
    
    printf("\n==============================================\n");
    printf("Done\n");
    printf("==============================================\n");
    
    return 0;
}


// ============================================================================
// Example 3: Advanced usage with filtering
// ============================================================================

/**
 * Example showing how to filter classifications
 */
class FilteredAudioProcessor : public ZODASAudioProcessor {
private:
    std::string last_class;
    int same_class_count;
    
public:
    FilteredAudioProcessor() : same_class_count(0) {}
    
    void OnAudioEventDetected(int class_id, 
                             const std::string& class_name,
                             float confidence) override {
        // Only report if:
        // 1. Confidence is high enough
        // 2. Same class detected multiple times (reduces false positives)
        
        if (class_name == last_class) {
            same_class_count++;
        } else {
            last_class = class_name;
            same_class_count = 1;
        }
        
        // Require 3 consecutive detections with confidence > 0.3
        if (same_class_count >= 3 && confidence > 0.3) {
            printf("[CONFIRMED] %s (%.2f)\n", class_name.c_str(), confidence);
            
            // Trigger your application logic here
            // ...
        }
    }
};


// ============================================================================
// Build Instructions
// ============================================================================

/*

To build this example:

g++ -std=c++17 -O3 \
  -I/path/to/tensorflow \
  -I/path/to/tensorflow/bazel-tensorflow/external/flatbuffers/include \
  zodas_integration_example.cpp \
  yamnet_classifier.cpp \
  -L/path/to/tensorflow/bazel-bin/tensorflow/lite \
  -ltensorflowlite \
  -ldl -lpthread \
  -o zodas_yamnet_example

To run:

export LD_LIBRARY_PATH=/path/to/tensorflow/bazel-bin/tensorflow/lite:$LD_LIBRARY_PATH
./zodas_yamnet_example

*/