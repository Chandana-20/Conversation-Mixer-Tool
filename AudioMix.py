from pydub import AudioSegment
from silero_vad import get_speech_timestamps, read_audio
import torch

# Detect Speech Segments with Silero VAD
def get_speech_segments(audio_path, sampling_rate=8000):
    # Load audio
    wav = read_audio(audio_path, sampling_rate=sampling_rate)
    
    # Load Silero VAD model and utilities
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        onnx=False,
        trust_repo=True
    )
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        wav, 
        model,
        sampling_rate=sampling_rate,
        return_seconds=True  # Return timestamps in seconds directly
    )
    
    # Convert timestamps to tuple format
    return [(ts['start'], ts['end']) for ts in speech_timestamps]

# Interleave Speech Segments
def interleave_segments(caller_segments, receiver_segments, caller_audio, receiver_audio):
    combined_audio = AudioSegment.silent(duration=0)
    i, j = 0, 0
    
    # Add small gap between segments (in milliseconds)
    gap_duration = 100
    gap = AudioSegment.silent(duration=gap_duration)
    
    while i < len(caller_segments) and j < len(receiver_segments):
        if caller_segments[i][0] < receiver_segments[j][0]:
            start, end = caller_segments[i]
            segment = caller_audio[int(start * 1000):int(end * 1000)]
            combined_audio += segment + gap
            i += 1
        else:
            start, end = receiver_segments[j]
            segment = receiver_audio[int(start * 1000):int(end * 1000)]
            combined_audio += segment + gap
            j += 1
            
    # Add remaining segments
    while i < len(caller_segments):
        start, end = caller_segments[i]
        segment = caller_audio[int(start * 1000):int(end * 1000)]
        combined_audio += segment + gap
        i += 1
        
    while j < len(receiver_segments):
        start, end = receiver_segments[j]
        segment = receiver_audio[int(start * 1000):int(end * 1000)]
        combined_audio += segment + gap
        j += 1
        
    return combined_audio

# Add Smooth Transitions
def smooth_transitions(audio, fade_duration=50):
    return audio.fade_in(fade_duration).fade_out(fade_duration)

# Main Function
def process_audio(caller_path, receiver_path, output_path):
    try:
        # Load filtered audio files
        caller_audio = AudioSegment.from_wav(caller_path)
        receiver_audio = AudioSegment.from_wav(receiver_path)
        
        print("Detecting speech segments in caller audio...")
        caller_segments = get_speech_segments(caller_path, sampling_rate=8000)
        
        print("Detecting speech segments in receiver audio...")
        receiver_segments = get_speech_segments(receiver_path, sampling_rate=8000)
        
        print("Interleaving segments...")
        combined_audio = interleave_segments(caller_segments, receiver_segments, caller_audio, receiver_audio)
        
        print("Adding smooth transitions...")
        combined_audio = smooth_transitions(combined_audio)
        
        # Export the output
        print(f"Exporting processed audio to {output_path}")
        combined_audio.export(output_path, format="wav")
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise

if __name__ == "__main__":
    # Update file paths as needed
    caller_file = "Audio/filtered_audio_Nandana.wav"
    receiver_file = "Audio/filtered_audio_Asha.wav"
    output_file = "Audio/conversation.wav"
    
    process_audio(caller_file, receiver_file, output_file)
