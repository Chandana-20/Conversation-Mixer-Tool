Noise Reduction using Spectral Subtraction
----------------------------------------
This script implements noise reduction using spectral subtraction technique.
It estimates the noise profile from a reference clean signal and removes
similar noise patterns from the noisy audio.

Features:
    - Noise profile estimation
    - Spectral subtraction
    - Audio length alignment
    - Spectrogram visualization
"""

import numpy as np
from scipy.io import wavfile
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display


def load_audio(file_path):
    """Load audio file and return sample rate and data"""
    sample_rate, audio_data = wavfile.read(file_path)
    # Convert to float32 for processing
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    return sample_rate, audio_data


def align_audio_lengths(noisy, clean):
    """
    Align the lengths of two audio arrays by zero-padding the shorter one 
    or trimming the longer one to match.
    
    Args:
        noisy (np.ndarray): Noisy audio array.
        clean (np.ndarray): Clean audio array.
    
    Returns:
        tuple: Aligned noisy and clean audio arrays.
    """
    len_noisy = len(noisy)
    len_clean = len(clean)
    
    if len_noisy > len_clean:
        print(f"Zero-padding clean audio from {len_clean} to {len_noisy} samples.")
        clean = np.pad(clean, (0, len_noisy - len_clean), mode='constant')
    elif len_clean > len_noisy:
        print(f"Zero-padding noisy audio from {len_noisy} to {len_clean} samples.")
        noisy = np.pad(noisy, (0, len_clean - len_noisy), mode='constant')
    else:
        print("No alignment needed. Audio lengths are already equal.")
    
    return noisy, clean


def get_noise_profile(noise_clip, frame_length=2048, hop_length=512):
    """Estimate noise profile from a noise clip"""
    # Compute spectrogram
    noise_stft = librosa.stft(noise_clip, n_fft=frame_length, hop_length=hop_length)
    noise_spec = np.abs(noise_stft)
    
    # Estimate noise profile
    noise_profile = np.mean(noise_spec, axis=1)
    return noise_profile


def spectral_subtraction(audio_data, noise_profile, frame_length=2048, hop_length=512, reduction_factor=1.0):
    """Apply spectral subtraction to remove noise"""
    # Compute STFT
    audio_stft = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
    audio_spec = np.abs(audio_stft)
    audio_angle = np.angle(audio_stft)
    
    # Reshape noise profile to match spectrogram
    noise_profile = noise_profile.reshape(-1, 1)
    
    # Subtract noise profile
    clean_spec = np.maximum(audio_spec - reduction_factor * noise_profile, 0)
    
    # Reconstruct signal
    clean_stft = clean_spec * np.exp(1j * audio_angle)
    clean_audio = librosa.istft(clean_stft, hop_length=hop_length)
    
    return clean_audio


def plot_spectrograms(original, cleaned, sample_rate):
    """Plot spectrograms of original and cleaned audio"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(
        np.abs(librosa.stft(original)), ref=np.max),
        sr=sample_rate, y_axis='hz')
    plt.title('Original Audio Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(
        np.abs(librosa.stft(cleaned)), ref=np.max),
        sr=sample_rate, y_axis='hz')
    plt.title('Cleaned Audio Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()


def main():
    # File paths
    noisy_file = 'Audio/Audoaudio.wav'
    clean_file = 'Audio/conversation.wav'
    output_file = 'cleaned_output.wav'
    
    # Load audio files
    sample_rate, noisy_audio = load_audio(noisy_file)
    _, clean_audio = load_audio(clean_file)
    
    # Align audio lengths
    noisy_audio, clean_audio = align_audio_lengths(noisy_audio, clean_audio)
    
    # Get noise profile (from the difference between noisy and clean audio)
    noise = noisy_audio - clean_audio
    noise_profile = get_noise_profile(noise)
    
    # Apply noise reduction
    cleaned_audio = spectral_subtraction(noisy_audio, noise_profile, reduction_factor=1.2)
    
    # Normalize audio
    cleaned_audio = cleaned_audio / np.max(np.abs(cleaned_audio))
    
    # Save cleaned audio
    sf.write(output_file, cleaned_audio, sample_rate)
    
    # Plot spectrograms
    plot_spectrograms(noisy_audio, cleaned_audio, sample_rate)
    
    print(f"Noise reduction complete. Cleaned audio saved to {output_file}")


if __name__ == "__main__":
    main()
