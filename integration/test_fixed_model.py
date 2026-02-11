"""Test the fixed YAMNet core model."""

import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import tensorflow_hub as hub

def compute_mel_patches(waveform):
    """Compute mel patches using 257 bins."""
    stft = tf.signal.stft(
        waveform,
        frame_length=400,
        frame_step=160,
        fft_length=512,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )
    
    magnitude = tf.abs(stft)
    
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=257,
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0
    )
    
    mel = tf.matmul(magnitude, mel_matrix)
    log_mel = tf.math.log(mel + 0.01)
    
    # Create patches with 50% overlap
    patches = []
    for i in range(0, log_mel.shape[0] - 96 + 1, 48):
        patch = log_mel[i:i+96, :]
        patches.append(patch)
    
    if len(patches) == 0:
        patch = tf.pad(log_mel, [[0, 96 - log_mel.shape[0]], [0, 0]])
        patches.append(patch)
    
    patches = tf.stack(patches)
    patches = tf.expand_dims(patches, -1)
    
    return patches

print("Loading models...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
core_model_loaded = tf.saved_model.load('yamnet_core')
core_model = core_model_loaded.signatures['serving_default']

print("\nLoading audio...")
sample_rate, wav_data = wavfile.read('wavs/miaow_16k.wav')
waveform = wav_data / 32767.0
waveform = waveform.astype(np.float32)

print(f"Audio: {waveform.shape}, {len(waveform)/16000:.2f}s")

print("\n" + "="*70)
print("YAMNET REFERENCE")
print("="*70)
scores_yamnet, _, mel_yamnet = yamnet(waveform)
print(f"Predictions shape: {scores_yamnet.shape}")
print(f"Output sum (first patch): {scores_yamnet[0].numpy().sum():.6f}")
print(f"Output range: [{scores_yamnet.numpy().min():.6f}, {scores_yamnet.numpy().max():.6f}]")
print(f"Top class: {scores_yamnet.numpy().mean(axis=0).argmax()}")

top_classes_yamnet = scores_yamnet.numpy().argmax(axis=1)
print(f"\nPer-patch top classes: {top_classes_yamnet}")

print("\n" + "="*70)
print("CORE MODEL (FIXED)")
print("="*70)
mel_patches = compute_mel_patches(waveform)
print(f"Mel patches: {mel_patches.shape}")

predictions_core_dict = core_model(mel_spectrogram=mel_patches)
predictions_core = predictions_core_dict['predictions'].numpy()
print(f"Predictions shape: {predictions_core.shape}")
print(f"Output sum (first patch): {predictions_core[0].sum():.6f}")
print(f"Output range: [{predictions_core.min():.6f}, {predictions_core.max():.6f}]")
print(f"Top class: {predictions_core.mean(axis=0).argmax()}")

top_classes_core = predictions_core.argmax(axis=1)
print(f"\nPer-patch top classes: {top_classes_core}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

# Compare using YAMNet's own mel
yamnet_mel_patches = []
for i in range(0, mel_yamnet.shape[0] - 96 + 1, 48):
    patch = mel_yamnet[i:i+96, :]
    yamnet_mel_patches.append(patch)

yamnet_mel_patches = tf.stack(yamnet_mel_patches)
yamnet_mel_patches = tf.expand_dims(yamnet_mel_patches, -1)

print(f"Using YAMNet's mel patches: {yamnet_mel_patches.shape}")
predictions_core_yamnet_mel_dict = core_model(mel_spectrogram=yamnet_mel_patches)
predictions_core_yamnet_mel = predictions_core_yamnet_mel_dict['predictions'].numpy()

print(f"\nCore model with YAMNet's mel:")
print(f"  Output sum (first): {predictions_core_yamnet_mel[0].sum():.6f}")
print(f"  Top class: {predictions_core_yamnet_mel.mean(axis=0).argmax()}")
print(f"  Per-patch top: {predictions_core_yamnet_mel.argmax(axis=1)}")

mae = np.mean(np.abs(scores_yamnet.numpy() - predictions_core_yamnet_mel))
print(f"\nMAE (YAMNet vs Core with same mel): {mae:.6f}")

matches = (top_classes_yamnet == predictions_core_yamnet_mel.argmax(axis=1)).sum()
print(f"Top class match rate: {matches}/{len(top_classes_yamnet)} ({100*matches/len(top_classes_yamnet):.1f}%)")

print("\n" + "="*70)
print("FRAME-BY-FRAME")
print("="*70)
for i in range(min(13, len(top_classes_yamnet))):
    core_class = predictions_core_yamnet_mel.argmax(axis=1)[i]
    match = "✓" if top_classes_yamnet[i] == core_class else "✗"
    print(f"Patch {i}: YAMNet={top_classes_yamnet[i]:3d}, Core={core_class:3d} {match}")