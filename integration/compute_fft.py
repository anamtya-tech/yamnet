"""
Python FFT helper for YAMNet C++ classifier.
Computes 257-bin magnitude spectra from waveform.
"""

import numpy as np
import tensorflow as tf
import sys
import struct

# YAMNet parameters
FRAME_LENGTH = 400  # 25ms at 16kHz
FRAME_STEP = 160    # 10ms at 16kHz
FFT_SIZE = 512

def compute_stft_magnitudes(waveform):
    """
    Compute STFT magnitude spectra from waveform.
    
    Args:
        waveform: 1D numpy array of float32 audio samples
        
    Returns:
        2D numpy array of shape (num_frames, 257) - magnitude spectra
    """
    # Compute STFT using TensorFlow (same as our working Python code)
    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_SIZE,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )
    
    # Get magnitude (257 bins)
    magnitude = tf.abs(stft).numpy()
    
    return magnitude


def read_wav_and_compute_spectra(wav_path):
    """
    Read WAV file and compute magnitude spectra.
    
    Args:
        wav_path: Path to 16kHz WAV file
        
    Returns:
        2D numpy array of shape (num_frames, 257)
    """
    from scipy.io import wavfile
    
    sample_rate, wav_data = wavfile.read(wav_path)
    
    if sample_rate != 16000:
        print(f"Warning: Sample rate is {sample_rate}Hz, expected 16000Hz", 
              file=sys.stderr)
    
    # Normalize to float32 [-1, 1]
    if wav_data.dtype == np.int16:
        waveform = wav_data.astype(np.float32) / 32767.0
    else:
        waveform = wav_data.astype(np.float32)
    
    # Compute spectra
    spectra = compute_stft_magnitudes(waveform)
    
    return spectra


def write_spectra_binary(spectra, output_path):
    """
    Write spectra to binary file for C++ to read.
    
    Format:
        int32: num_frames
        int32: num_bins (should be 257)
        float32[num_frames * num_bins]: spectra data (row-major)
    """
    num_frames, num_bins = spectra.shape
    
    with open(output_path, 'wb') as f:
        # Write dimensions
        f.write(struct.pack('i', num_frames))
        f.write(struct.pack('i', num_bins))
        
        # Write data
        spectra.astype(np.float32).tofile(f)
    
    print(f"Wrote {num_frames} frames x {num_bins} bins to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute STFT magnitudes for YAMNet')
    parser.add_argument('input_wav', help='Input WAV file (16kHz)')
    parser.add_argument('output_bin', help='Output binary file')
    
    args = parser.parse_args()
    
    # Compute and save
    spectra = read_wav_and_compute_spectra(args.input_wav)
    write_spectra_binary(spectra, args.output_bin)
    
    print(f"Shape: {spectra.shape}")
    print(f"Range: [{spectra.min():.6f}, {spectra.max():.6f}]")