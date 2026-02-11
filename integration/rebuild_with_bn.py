"""Rebuild YAMNet core with COMPLETE weight transfer including BN states."""

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
    """YAMNet core with mel input."""
    mel_input = layers.Input(shape=(96, 64, 1), name='mel_spectrogram')
    
    net = mel_input
    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
        net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)
    
    embeddings = layers.GlobalAveragePooling2D(name='embeddings')(net)
    logits = layers.Dense(units=params.num_classes, use_bias=True, name='logits')(embeddings)
    predictions = layers.Activation(activation='sigmoid', name='predictions')(logits)
    
    model = Model(inputs=mel_input, outputs=predictions, name='yamnet_core')
    return model

print("Loading YAMNet from TFHub...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

print("Creating params...")
params = params_module.Params()

print("Creating core model...")
core_model = yamnet_core_model(params)

# Build both models
print("\nBuilding models...")
dummy_waveform = tf.zeros((15600,), dtype=tf.float32)
dummy_mel = tf.zeros((1, 96, 64, 1), dtype=tf.float32)

_ = yamnet(dummy_waveform)
_ = core_model(dummy_mel)

print("\n" + "="*70)
print("TRANSFERRING ALL VARIABLES (trainable + non-trainable)")
print("="*70)

# Get ALL variables (including BN moving mean/variance)
yamnet_all_vars = yamnet._yamnet.variables
core_all_vars = core_model.variables

print(f"\nYAMNet total variables: {len(yamnet_all_vars)}")
print(f"Core total variables: {len(core_all_vars)}")

if len(yamnet_all_vars) != len(core_all_vars):
    print(f"⚠ WARNING: Variable count mismatch!")
else:
    print(f"✓ Variable counts match!")

transferred = 0
mismatched = 0

for i, (yv, cv) in enumerate(zip(yamnet_all_vars, core_all_vars)):
    if yv.shape == cv.shape:
        cv.assign(yv)
        transferred += 1
        # Print first/last few and all BN moving stats
        if i < 3 or i >= len(core_all_vars) - 2 or 'moving' in cv.name:
            trainable = "T" if cv.trainable else "NT"
            print(f"  {trainable} {i:3d}: {cv.name[:60]:60s} {str(cv.shape):20s}")
    else:
        print(f"  ✗ {i}: Shape mismatch - yamnet: {yv.shape}, core: {cv.shape}")
        mismatched += 1

print(f"\n✓ Transferred: {transferred}/{len(core_all_vars)}")
print(f"✗ Mismatched: {mismatched}")

# Verify a BN moving mean was transferred
print("\n" + "="*70)
print("VERIFICATION: Batch Norm States")
print("="*70)

bn_vars_yamnet = [v for v in yamnet_all_vars if 'moving_mean' in v.name]
bn_vars_core = [v for v in core_all_vars if 'moving_mean' in v.name]

if bn_vars_core:
    print(f"\nFirst BN moving_mean:")
    print(f"  YAMNet: {bn_vars_yamnet[0].numpy()[:5]}")
    print(f"  Core:   {bn_vars_core[0].numpy()[:5]}")
    print(f"  Match: {np.allclose(bn_vars_yamnet[0].numpy(), bn_vars_core[0].numpy())}")

# Test inference
print("\n" + "="*70)
print("TESTING INFERENCE")
print("="*70)

test_input = tf.random.normal((1, 96, 64, 1))
core_pred = core_model(test_input).numpy()

print(f"Test output sum: {core_pred.sum():.6f}")
print(f"Test output range: [{core_pred.min():.6f}, {core_pred.max():.6f}]")

if core_pred.sum() > 10:
    print("\n✗ FAILED: Still broken!")
else:
    print("\n✓ SUCCESS: Model works correctly!")
    
    # Save
    print("\nSaving model...")
    core_model.save('yamnet_core')
    print("✓ Saved to yamnet_core")
    
    # Verify saved model
    print("\nVerifying saved model...")
    reloaded = tf.saved_model.load('yamnet_core')
    reloaded_pred = reloaded.signatures['serving_default'](mel_spectrogram=test_input)['predictions'].numpy()
    print(f"Reloaded output sum: {reloaded_pred.sum():.6f}")
    
    if reloaded_pred.sum() < 10:
        print("✓✓✓ PERFECT! Saved model works!")
    else:
        print("✗ Saved model is broken again")