/**
 * @file yamnet_classifier.h
 * @brief YAMNet audio classification for ZODAS integration
 * 
 * This header provides a simple API for real-time audio classification
 * using Google's YAMNet model. Designed for integration with ZODAS
 * audio processing pipeline.
 * 
 * @author [Your Name]
 * @date 2025-12-20
 * @version 1.0
 */

#ifndef YAMNET_CLASSIFIER_H
#define YAMNET_CLASSIFIER_H

#include <string>
#include <vector>
#include <memory>

/**
 * @class YAMNetClassifier
 * @brief Real-time audio event classification using YAMNet
 * 
 * This class provides frame-by-frame audio classification. It buffers
 * 257-bin magnitude spectra from ZODAS and performs classification
 * when sufficient frames are accumulated.
 * 
 * @example Basic Usage
 * @code
 * YAMNetClassifier classifier;
 * classifier.LoadModel("yamnet_core.tflite");
 * classifier.LoadClassNames("yamnet_class_map.csv");
 * 
 * // In your audio callback (called every 10ms)
 * void audio_callback(float* spectrum_257bins) {
 *     int class_id;
 *     std::string class_name;
 *     float confidence;
 *     
 *     bool ready = classifier.AddFrame(spectrum_257bins, 
 *                                      class_id, class_name, confidence);
 *     if (ready) {
 *         printf("Detected: %s\n", class_name.c_str());
 *     }
 * }
 * @endcode
 */
class YAMNetClassifier {
public:
    /**
     * @brief Constructor
     */
    YAMNetClassifier();
    
    /**
     * @brief Destructor
     */
    ~YAMNetClassifier();
    
    // ========================================================================
    // Initialization (call once at startup)
    // ========================================================================
    
    /**
     * @brief Load the TFLite model
     * 
     * @param tflite_model_path Path to yamnet_core.tflite file
     * @return true if model loaded successfully, false otherwise
     * 
     * @note Must be called before AddFrame()
     */
    bool LoadModel(const char* tflite_model_path);
    
    /**
     * @brief Load class name mappings
     * 
     * @param csv_path Path to yamnet_class_map.csv file
     * @return true if class names loaded successfully, false otherwise
     * 
     * @note Must be called before AddFrame()
     */
    bool LoadClassNames(const char* csv_path);
    
    // ========================================================================
    // Frame-by-Frame Processing (real-time API)
    // ========================================================================
    
    /**
     * @brief Add a new spectrum frame and get classification when ready
     * 
     * This is the main function for real-time processing. Call it for each
     * spectrum frame from ZODAS (every 10ms). It will return true when a
     * new classification is available.
     * 
     * @param magnitude_spectrum_257bins Input spectrum from ZODAS (257 floats)
     * @param class_id_out Output: predicted class ID (0-520)
     * @param class_name_out Output: predicted class name (e.g., "Speech")
     * @param confidence_out Output: confidence score (0.0-1.0)
     * @return true if new classification is ready, false if still buffering
     * 
     * @note Buffering behavior:
     *   - First 96 frames: returns false (buffering)
     *   - Frame 96: returns true (first classification)
     *   - Subsequently: returns true every 48 frames (50% overlap)
     * 
     * @warning magnitude_spectrum_257bins must contain exactly 257 values!
     *          Using 128 bins will produce incorrect results.
     * 
     * @example
     * @code
     * float spectrum[257];
     * zodas_get_spectrum(spectrum);  // Your ZODAS API
     * 
     * int class_id;
     * std::string class_name;
     * float confidence;
     * 
     * if (classifier.AddFrame(spectrum, class_id, class_name, confidence)) {
     *     // New classification ready
     *     if (confidence > 0.5) {
     *         printf("High confidence: %s\n", class_name.c_str());
     *     }
     * }
     * @endcode
     */
    bool AddFrame(const float* magnitude_spectrum_257bins,
                  int& class_id_out,
                  std::string& class_name_out,
                  float& confidence_out);
    
    /**
     * @brief Reset the frame buffer
     * 
     * Call this when starting a new audio stream or after a discontinuity.
     * Clears all buffered frames and resets to initial state.
     */
    void Reset();
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    /**
     * @brief Get class name from class ID
     * 
     * @param class_id Class ID (0-520)
     * @return Class name string, or "Unknown" if ID is invalid
     */
    std::string GetClassName(int class_id) const;
    
    /**
     * @brief Get number of classes
     * 
     * @return Number of classes (should be 521)
     */
    int GetNumClasses() const;
    
    /**
     * @brief Check if model is loaded and ready
     * 
     * @return true if model is loaded and ready for inference
     */
    bool IsReady() const;
    
    /**
     * @brief Print model information (for debugging)
     */
    void PrintModelInfo() const;
    
private:
    // Implementation details hidden in .cpp file
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Disable copy and assignment
    YAMNetClassifier(const YAMNetClassifier&) = delete;
    YAMNetClassifier& operator=(const YAMNetClassifier&) = delete;
};

// ============================================================================
// Constants
// ============================================================================

namespace YAMNet {
    /**
     * @brief YAMNet model parameters
     */
    namespace Params {
        constexpr int SAMPLE_RATE = 16000;        ///< Required sample rate (Hz)
        constexpr int FRAME_LENGTH = 400;         ///< Frame length in samples (25ms)
        constexpr int FRAME_STEP = 160;           ///< Frame hop in samples (10ms)
        constexpr int FFT_SIZE = 512;             ///< FFT size
        constexpr int SPECTRUM_BINS = 257;        ///< Number of spectrum bins required
        constexpr int MEL_BINS = 64;              ///< Number of mel bins
        constexpr int PATCH_FRAMES = 96;          ///< Frames per classification
        constexpr int PATCH_HOP = 48;             ///< Frame hop between classifications
        constexpr int NUM_CLASSES = 521;          ///< Number of output classes
        constexpr float MEL_MIN_HZ = 125.0f;      ///< Mel filterbank min frequency
        constexpr float MEL_MAX_HZ = 7500.0f;     ///< Mel filterbank max frequency
        constexpr float LOG_OFFSET = 0.001f;      ///< Log stabilization offset
    }
    
    /**
     * @brief Common audio event class IDs
     * 
     * These are some frequently used classes. See yamnet_class_map.csv
     * for the complete list of 521 classes.
     */
    namespace ClassID {
        constexpr int SPEECH = 0;
        constexpr int MALE_SPEECH = 1;
        constexpr int FEMALE_SPEECH = 2;
        constexpr int CHILD_SPEECH = 3;
        constexpr int CONVERSATION = 4;
        constexpr int MUSIC = 137;
        constexpr int SINGING = 176;
        constexpr int ANIMAL = 67;
        constexpr int CAT = 76;
        constexpr int DOG = 74;
        constexpr int BIRD = 71;
        constexpr int VEHICLE = 88;
        constexpr int CAR = 89;
        constexpr int ALARM = 388;
        constexpr int DOOR = 394;
        constexpr int FOOTSTEPS = 402;
        constexpr int LAUGHTER = 321;
        constexpr int APPLAUSE = 348;
        constexpr int SILENCE = 486;
    }
}

#endif // YAMNET_CLASSIFIER_H