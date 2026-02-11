"""Correct mel spectrogram computation matching YAMNet exactly."""

import tensorflow as tf

def compute_mel_spectrogram_yamnet(waveform):
    """
    Compute mel spectrogram patches exactly as YAMNet does.
    
    Parameters from YAMNet:
    - frame_length: 400 samples (25ms at 16kHz)
    - frame_step: 160 samples (10ms at 16kHz)
    - fft_length: 512
    - mel_bins: 64
    - mel_min_hz: 125.0
    - mel_max_hz: 7500.0
    - log_offset: 0.001  <-- CRITICAL: This was 0.01 before!
    
    Args:
        waveform: 1D tensor of audio samples at 16kHz
        
    Returns:
        Tensor of shape (num_patches, 96, 64, 1) ready for YAMNet core model
    """
    # STFT computation
    stft = tf.signal.stft(
        waveform,
        frame_length=400,
        frame_step=160,
        fft_length=512,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )
    
    # Magnitude spectrogram (257 frequency bins)
    magnitude = tf.abs(stft)
    
    # Mel filterbank: 257 FFT bins → 64 mel bins
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=257,
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0
    )
    
    # Apply mel filterbank
    mel = tf.matmul(magnitude, mel_matrix)
    
    # Log mel spectrogram with correct offset
    log_mel = tf.math.log(mel + 0.001)  # ← Changed from 0.01 to 0.001
    
    # Create overlapping 96-frame patches (50% overlap, hop=48 frames)
    patch_frames = 96
    patch_hop = 48
    
    patches = []
    for i in range(0, log_mel.shape[0] - patch_frames + 1, patch_hop):
        patch = log_mel[i:i+patch_frames, :]
        patches.append(patch)
    
    # Handle short audio (pad if needed)
    if len(patches) == 0:
        patch = tf.pad(log_mel, [[0, patch_frames - log_mel.shape[0]], [0, 0]])
        patches.append(patch)
    
    # Stack and add channel dimension
    patches = tf.stack(patches)
    patches = tf.expand_dims(patches, -1)
    
    return patches


# Quick test
if __name__ == '__main__':
    import numpy as np
    from scipy.io import wavfile
    import tensorflow_hub as hub
    
    print("Testing corrected mel computation...")
    
    # Load audio
    sample_rate, wav_data = wavfile.read('wavs/miaow_16k.wav')
    waveform = wav_data / 32767.0
    waveform = waveform.astype(np.float32)
    
    # Load YAMNet for comparison
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    _, _, mel_yamnet = yamnet(waveform)
    
    # Our computation
    our_patches = compute_mel_spectrogram_yamnet(waveform)
    
    # Extract first patch for comparison
    our_first_patch = our_patches[0, :, :, 0].numpy()
    yamnet_first_patch = mel_yamnet[:96, :].numpy()
    
    mae = np.mean(np.abs(our_first_patch - yamnet_first_patch))
    max_diff = np.abs(our_first_patch - yamnet_first_patch).max()
    
    print(f"\nFirst patch comparison:")
    print(f"  MAE: {mae:.9f}")
    print(f"  Max diff: {max_diff:.9f}")
    
    if mae < 0.00001:
        print("  ✓✓✓ PERFECT MATCH!")
    elif mae < 0.001:
        print("  ✓ Very close match")
    else:
        print("  ✗ Still has differences")
    
    print(f"\nOur patches shape: {our_patches.shape}")
    print(f"YAMNet mel shape: {mel_yamnet.shape}")