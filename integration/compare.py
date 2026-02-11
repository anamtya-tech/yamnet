"""Compare mel spectrograms between YAMNet and custom computation."""

import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import tensorflow_hub as hub

# Load YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def yamnet_mel_computation(waveform):
    """Get YAMNet's mel spectrogram."""
    scores, embeddings, mel = yamnet_model(waveform)
    return mel

def custom_mel_from_257bins(waveform):
    """Compute mel using YAMNet's actual FFT size (257 bins from 512 FFT)."""
    # YAMNet parameters
    frame_length = 400
    frame_step = 160
    fft_length = 512
    
    # Compute STFT on entire waveform (no batching)
    stft = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )
    
    magnitude = tf.abs(stft)  # Shape: [num_frames, 257]
    
    # Create mel filterbank (257 → 64)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=257,
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0
    )
    
    mel = tf.matmul(magnitude, mel_matrix)
    log_mel = tf.math.log(mel + 0.01)
    
    # Now batch into 96-frame chunks for model input
    num_frames = tf.shape(log_mel)[0]
    num_batches = num_frames // 96
    
    # Trim to complete batches
    log_mel_batched = log_mel[:num_batches * 96, :]
    log_mel_batched = tf.reshape(log_mel_batched, [num_batches, 96, 64])
    log_mel_batched = tf.expand_dims(log_mel_batched, -1)
    
    return log_mel, log_mel_batched  # Return both full and batched versions

def custom_mel_from_128bins(waveform):
    """Your current computation using 128 bins (ZODAS)."""
    frame_length = 400
    frame_step = 160
    fft_length = 512
    
    # Compute STFT on entire waveform
    stft = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )
    
    magnitude = tf.abs(stft)[:, :128]  # Only first 128 bins
    
    # Mel filterbank (128 → 64)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=128,
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0
    )
    
    mel = tf.matmul(magnitude, mel_matrix)
    log_mel = tf.math.log(mel + 0.01)
    
    # Batch into 96-frame chunks
    num_frames = tf.shape(log_mel)[0]
    num_batches = num_frames // 96
    
    log_mel_batched = log_mel[:num_batches * 96, :]
    log_mel_batched = tf.reshape(log_mel_batched, [num_batches, 96, 64])
    log_mel_batched = tf.expand_dims(log_mel_batched, -1)
    
    return log_mel, log_mel_batched

if __name__ == '__main__':
    # Load audio
    sample_rate, wav_data = wavfile.read('wavs/miaow_16k.wav')
    waveform = wav_data / 32767.0
    waveform = waveform.astype(np.float32)
    
    print(f"Waveform shape: {waveform.shape}")
    print(f"Duration: {len(waveform) / 16000:.2f}s\n")
    
    # Get all three versions
    yamnet_mel = yamnet_mel_computation(waveform)
    custom_257_full, custom_257_batched = custom_mel_from_257bins(waveform)
    custom_128_full, custom_128_batched = custom_mel_from_128bins(waveform)
    
    print("="*60)
    print("MEL SPECTROGRAM COMPARISON")
    print("="*60)
    
    print(f"\nYAMNet mel:            {yamnet_mel.shape}")
    print(f"Custom (257 bins) full: {custom_257_full.shape}")
    print(f"Custom (257 bins) batched: {custom_257_batched.shape}")
    print(f"Custom (128 bins) full: {custom_128_full.shape}")
    print(f"Custom (128 bins) batched: {custom_128_batched.shape}")
    
    print(f"\nYAMNet mel range:       [{yamnet_mel.numpy().min():.3f}, {yamnet_mel.numpy().max():.3f}]")
    print(f"Custom (257) range:     [{custom_257_full.numpy().min():.3f}, {custom_257_full.numpy().max():.3f}]")
    print(f"Custom (128) range:     [{custom_128_full.numpy().min():.3f}, {custom_128_full.numpy().max():.3f}]")
    
    # Compare first frame
    print("\n" + "="*60)
    print("FIRST FRAME COMPARISON (Frame 0)")
    print("="*60)
    
    # YAMNet output is (num_frames, 64), custom is (num_frames, 64)
    yamnet_frame0 = yamnet_mel[0, :].numpy()  # Shape: (64,)
    custom_257_frame0 = custom_257_full[0, :].numpy()  # Shape: (64,)
    custom_128_frame0 = custom_128_full[0, :].numpy()  # Shape: (64,)
    
    mae_257 = np.mean(np.abs(yamnet_frame0 - custom_257_frame0))
    mae_128 = np.mean(np.abs(yamnet_frame0 - custom_128_frame0))
    
    print(f"\nMean Absolute Error (257 bins): {mae_257:.6f}")
    print(f"Mean Absolute Error (128 bins): {mae_128:.6f}")
    
    corr_257 = np.corrcoef(yamnet_frame0, custom_257_frame0)[0, 1]
    corr_128 = np.corrcoef(yamnet_frame0, custom_128_frame0)[0, 1]
    
    print(f"\nCorrelation (257 bins): {corr_257:.6f}")
    print(f"Correlation (128 bins): {corr_128:.6f}")
    
    # Also compare entire first batch (96 frames)
    print("\n" + "="*60)
    print("FIRST BATCH COMPARISON (96 frames)")
    print("="*60)
    
    yamnet_batch0 = yamnet_mel[:96, :].numpy()  # Shape: (96, 64)
    custom_257_batch0 = custom_257_batched[0, :, :, 0].numpy()  # Shape: (96, 64)
    custom_128_batch0 = custom_128_batched[0, :, :, 0].numpy()  # Shape: (96, 64)
    
    mae_257_batch = np.mean(np.abs(yamnet_batch0 - custom_257_batch0))
    mae_128_batch = np.mean(np.abs(yamnet_batch0 - custom_128_batch0))
    
    print(f"\nMean Absolute Error (257 bins): {mae_257_batch:.6f}")
    print(f"Mean Absolute Error (128 bins): {mae_128_batch:.6f}")
    
    print(f"\nMax difference (257 bins): {np.abs(yamnet_batch0 - custom_257_batch0).max():.6f}")
    print(f"Max difference (128 bins): {np.abs(yamnet_batch0 - custom_128_batch0).max():.6f}")
    
    # Compare all frames
    print("\n" + "="*60)
    print("ALL FRAMES COMPARISON")
    print("="*60)
    
    num_frames_compare = min(yamnet_mel.shape[0], custom_257_full.shape[0])
    yamnet_all = yamnet_mel[:num_frames_compare, :].numpy()
    custom_257_all = custom_257_full[:num_frames_compare, :].numpy()
    custom_128_all = custom_128_full[:num_frames_compare, :].numpy()
    
    mae_257_all = np.mean(np.abs(yamnet_all - custom_257_all))
    mae_128_all = np.mean(np.abs(yamnet_all - custom_128_all))
    
    print(f"\nComparing {num_frames_compare} frames")
    print(f"Mean Absolute Error (257 bins): {mae_257_all:.6f}")
    print(f"Mean Absolute Error (128 bins): {mae_128_all:.6f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    if mae_257 < mae_128:
        print("✓ Use 257 FFT bins (full STFT) for better accuracy")
        print("  Your ZODAS output should provide all 257 bins, not just 128")
    else:
        print("  128 bins is acceptable, but check if ZODAS can give 257 bins")