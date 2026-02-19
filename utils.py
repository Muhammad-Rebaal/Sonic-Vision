import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import librosa
import io
import soundfile as sf

class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=44100,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()
        )    
    
    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)  
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)

def process_uploaded_audio(audio_bytes):
    """
    Reads audio bytes, resamples to 44100Hz, and converts to mono.
    Returns:
        audio_data (np.array): The processed audio waveform.
        sample_rate (int): The sample rate (always 44100).
    """
    # Read the audio bytes
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    
    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample to 44100Hz if needed
    if sample_rate != 44100:
        audio_data = librosa.resample(
            y=audio_data, orig_sr=sample_rate,
            target_sr=44100
        )
        sample_rate = 44100
        
    return audio_data, sample_rate
