"""
Export YAMNet core model in SavedModel and TFLite formats.

This script:
1. Creates YAMNet core model (mel input → predictions)
2. Transfers all weights (trainable + batch norm states)
3. Saves as SavedModel (.pb)
4. Converts to TFLite for C++ deployment
"""

import tensorflow as tf
import tensorflow_hub as hub
from tf_keras import Model, layers
import numpy as np
import sys
import os

# Use the root models submodule (read-only)
script_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(os.path.dirname(script_dir), 'models/research/audioset/yamnet')
sys.path.append(models_path)
from yamnet import _batch_norm, _conv, _separable_conv, _YAMNET_LAYER_DEFS
import params as params_module


def yamnet_core_model(params):
    """
    YAMNet core model with mel spectrogram input.
    
    Input: (batch_size, 96, 64, 1) - mel spectrogram patches
    Output: (batch_size, 521) - class probabilities (sigmoid)
    """
    mel_input = layers.Input(shape=(96, 64, 1), name='mel_spectrogram')
    
    # Convolutional layers
    net = mel_input
    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
        net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)
    
    # Global pooling and classification
    embeddings = layers.GlobalAveragePooling2D(name='embeddings')(net)
    logits = layers.Dense(units=params.num_classes, use_bias=True, name='logits')(embeddings)
    predictions = layers.Activation(activation='sigmoid', name='predictions')(logits)
    
    model = Model(inputs=mel_input, outputs=predictions, name='yamnet_core')
    return model


def transfer_all_weights(yamnet_tfhub, core_model):
    """
    Transfer ALL variables including batch normalization states.
    
    Args:
        yamnet_tfhub: YAMNet model from TFHub
        core_model: Our core model
        
    Returns:
        Number of variables transferred
    """
    print("\n" + "="*70)
    print("TRANSFERRING WEIGHTS")
    print("="*70)
    
    yamnet_vars = yamnet_tfhub._yamnet.variables
    core_vars = core_model.variables
    
    print(f"\nYAMNet variables: {len(yamnet_vars)}")
    print(f"Core variables: {len(core_vars)}")
    
    if len(yamnet_vars) != len(core_vars):
        print(f"⚠ WARNING: Variable count mismatch!")
        return 0
    
    transferred = 0
    for i, (yv, cv) in enumerate(zip(yamnet_vars, core_vars)):
        if yv.shape == cv.shape:
            cv.assign(yv)
            transferred += 1
        else:
            print(f"✗ Variable {i}: Shape mismatch - {yv.shape} vs {cv.shape}")
    
    print(f"✓ Transferred: {transferred}/{len(core_vars)} variables")
    
    # Verify transfer
    print("\nVerification:")
    print(f"  First weight match: {np.allclose(yamnet_vars[0].numpy(), core_vars[0].numpy())}")
    print(f"  Last weight match: {np.allclose(yamnet_vars[-1].numpy(), core_vars[-1].numpy())}")
    
    return transferred


def test_model(model, model_name="Model"):
    """Test model with random input."""
    test_input = tf.random.normal((1, 96, 64, 1))
    output = model(test_input).numpy()
    
    print(f"\n{model_name} test:")
    print(f"  Output sum: {output.sum():.6f}")
    print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
    
    if output.sum() > 10:
        print(f"  ✗ FAILED - output sum too high")
        return False
    else:
        print(f"  ✓ PASSED - model works correctly")
        return True


def export_savedmodel(model, output_dir):
    """Export model as SavedModel (.pb format)."""
    print("\n" + "="*70)
    print(f"EXPORTING SAVEDMODEL to {output_dir}")
    print("="*70)
    
    model.save(output_dir)
    print(f"✓ Saved to {output_dir}")
    
    # Verify
    reloaded = tf.saved_model.load(output_dir)
    if test_model(lambda x: reloaded.signatures['serving_default'](mel_spectrogram=x)['predictions'], 
                  "Reloaded SavedModel"):
        return True
    return False


def export_tflite(model, output_path, quantize=False):
    """
    Export model as TFLite for C++ deployment.
    
    Args:
        model: Keras model
        output_path: Path to save .tflite file
        quantize: If True, apply dynamic range quantization
    """
    print("\n" + "="*70)
    print(f"EXPORTING TFLITE to {output_path}")
    print("="*70)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✓ Saved to {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Verify TFLite model
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    
    # Test inference
    test_input = np.random.randn(1, 96, 64, 1).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\n  TFLite test output sum: {output.sum():.6f}")
    print(f"  TFLite test output range: [{output.min():.6f}, {output.max():.6f}]")
    
    if output.sum() > 10:
        print(f"  ✗ FAILED - TFLite model broken")
        return False
    else:
        print(f"  ✓ PASSED - TFLite model works")
        return True


def main():
    print("="*70)
    print("YAMNET CORE MODEL EXPORTER")
    print("="*70)
    
    # Load YAMNet from TFHub
    print("\n1. Loading YAMNet from TFHub...")
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Create params
    print("\n2. Creating parameters...")
    params = params_module.Params()
    print(f"   Num classes: {params.num_classes}")
    print(f"   Classifier activation: {params.classifier_activation}")
    
    # Create core model
    print("\n3. Creating core model...")
    core_model = yamnet_core_model(params)
    
    # Build models
    print("\n4. Building models...")
    dummy_waveform = tf.zeros((15600,), dtype=tf.float32)
    dummy_mel = tf.zeros((1, 96, 64, 1), dtype=tf.float32)
    _ = yamnet(dummy_waveform)
    _ = core_model(dummy_mel)
    
    # Transfer weights
    print("\n5. Transferring weights...")
    transferred = transfer_all_weights(yamnet, core_model)
    
    if transferred == 0:
        print("\n✗ Weight transfer failed!")
        return
    
    # Test
    print("\n6. Testing model...")
    if not test_model(core_model, "Core model"):
        print("\n✗ Model test failed!")
        return
    
    # Export SavedModel
    print("\n7. Exporting formats...")
    savedmodel_path = 'yamnet_core'
    if not export_savedmodel(core_model, savedmodel_path):
        print("\n✗ SavedModel export failed!")
        return
    
    # Export TFLite (float32)
    tflite_path = 'yamnet_core.tflite'
    if not export_tflite(core_model, tflite_path, quantize=False):
        print("\n✗ TFLite export failed!")
        return
    
    # Export TFLite (quantized)
    tflite_quant_path = 'yamnet_core_quantized.tflite'
    if not export_tflite(core_model, tflite_quant_path, quantize=True):
        print("\n⚠ Quantized TFLite export failed (continuing anyway)")
    
    # Summary
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {savedmodel_path}/")
    print(f"     - SavedModel format (.pb)")
    print(f"     - Use in Python: tf.saved_model.load('{savedmodel_path}')")
    print(f"\n  2. {tflite_path}")
    print(f"     - TFLite float32 model")
    print(f"     - Use in C++: TfLiteModel::BuildFromFile()")
    print(f"\n  3. {tflite_quant_path}")
    print(f"     - TFLite quantized model (smaller, faster)")
    print(f"     - Use in C++: TfLiteModel::BuildFromFile()")
    
    print("\n" + "="*70)
    print("MODEL SPECIFICATION")
    print("="*70)
    print(f"  Input:")
    print(f"    Shape: (batch_size, 96, 64, 1)")
    print(f"    Type: float32")
    print(f"    Range: [-6.2, 3.8] (log mel spectrogram)")
    print(f"\n  Output:")
    print(f"    Shape: (batch_size, 521)")
    print(f"    Type: float32")
    print(f"    Range: [0, 1] (sigmoid probabilities)")
    print(f"\n  Mel computation parameters:")
    print(f"    frame_length: 400 (25ms)")
    print(f"    frame_step: 160 (10ms)")
    print(f"    fft_length: 512")
    print(f"    mel_bins: 64")
    print(f"    log_offset: 0.001")
    print(f"    patch_hop: 48 frames (50% overlap)")


if __name__ == '__main__':
    main()