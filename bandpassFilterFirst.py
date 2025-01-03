import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import librosa

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to the audio data.

    Args:
        data: Audio signal as a NumPy array.
        lowcut: Lower frequency cutoff in Hz.
        highcut: Higher frequency cutoff in Hz.
        fs: Sampling rate in Hz.
        order: Order of the filter (default is 5).

    Returns:
        Filtered audio signal as a NumPy array.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if not (0 < low < high < 1):
        raise ValueError(f"Cutoff frequencies must satisfy 0 < low < high < Nyquist. Got low={low:.4f}, high={high:.4f}")
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data

# Parameters
lowcut = 50.0    # Lower cutoff frequency (Hz)
highcut = 3999.0 # Slightly below Nyquist frequency (Hz)
input_file ="Audio/commercial-aircraft-pre-flight-instructions-56658.wav"  # Input audio file path
output_file = "filtered_audio_airplaneAirhostess.wav"    # Output file path

# Load audio file
print("Loading audio file...")
audio, sr = librosa.load(input_file, sr=None)  # Preserve original sampling rate

# Apply bandpass filter
print(f"Applying bandpass filter (lowcut={lowcut} Hz, highcut={highcut} Hz, sr={sr} Hz)...")
filtered_audio = bandpass_filter(audio, lowcut, highcut, sr)

# Save filtered audio
print(f"Saving filtered audio to {output_file}...")
sf.write(output_file, filtered_audio, sr)

print("Filtering complete!")
