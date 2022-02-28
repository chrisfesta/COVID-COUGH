import os
import librosa
import numpy as np
import random
import io
import pandas as pd
from Dataset import SingleInputGenerator


class AudioProcessing():
    """
    Load an audio file and process it for training
    """

    def __init__(self, file_path, resize_to=0):
        assert isinstance(file_path, io.BytesIO), 'File is not io.BytesIO type'
        self.file_path = file_path
        self._open()
        if resize_to > 0:
            self.resize(max_ms=resize_to)

    def _open(self):
        """
        Open audio file and store the variables: 'audio' and 'sr'
        """
        self.aud = librosa.load(self.file_path)
        audio, sr = self.aud
        self.audio = audio
        self.sr = sr
        self.length = librosa.get_duration(y=self.audio, sr=self.sr)

    def get_metadata(self):
        """
        Return: uuid, file_path, sr, length
        """
        return [self.file_path, self.sr, self.length]

    def is_stereo(self):
        return True if self.audio.ndim == 2 else False

    def rechannel(self, signal_type='stereo', audio=None):
        """
        Convert audio to the number of channels; this is used to convert mono to stereo or stereo to mono
        Args:
          signal_type: str
            The type of signal. Auto is stereo; stereo has 2 channels and mono has 1
          audio: ndarray
            The audio signal to convert to stereo or mono; default is none and function will use the class object instead

        """
        assert signal_type in [
            'auto', 'stereo', 'mono'], 'signal_type must be auto, stereo, or mono'
        assert isinstance(signal_type, str), 'signal_type is not a str'

        if signal_type == 'auto':
            n_channels = 2
        elif signal_type == 'stereo':
            n_channels = 2
        elif signal_type == 'mono':
            n_channels = 1

        # convert the class audio
        if audio is None:
            # already at desired number of channels; nothing to do
            if self.audio.ndim == n_channels:
                pass
            # Convert from stereo to mono by selecting only the first channel
            elif n_channels == 1:
                self.audio = librosa.to_mono(self.audio)
            # Convert from mono to stero by duplicating the first channel
            else:
                self.audio = np.asfortranarray(
                    np.array([self.audio, self.audio]))
        # conver the passed argument audio
        else:
            # already at desired number of channels; nothing to do
            if audio.ndim == n_channels:
                return audio
            # Convert from stereo to mono by selecting only the first channel
            elif n_channels == 1:
                return librosa.to_mono(audio)
            # Convert from mono to stero by duplicating the first channel
            else:
                return np.asfortranarray(np.array([audio, audio]))

    def resample(self, new_sr):
        """
        Resample audio to a new sample rate
        Args:
          new_sr: int
            The signal rate to resample the audio file to
        """
        # nothing to do, audio already at the new_sr
        if self.sr == new_sr:
            pass
        # resample the audio to the new_sr
        else:
            librosa.resample(y=self.audio, orig_sr=self.sr, target_sr=new_sr)

    def resize(self, max_ms):
        """
        Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
        Args:
          max_ms: int
            a fixed length 'max_ms' in milliseconds
        """
        audio_shape_len = len(self.audio.shape)

        # mono
        if audio_shape_len == 1:
            num_rows = 1
            sig_len = self.audio.shape[0]
        # stereo
        else:
            num_rows, sig_len = self.audio.shape
        max_len = int(self.sr // 1000 * max_ms)

        # Truncate the signal to the given length
        if sig_len > max_len:
            # mono
            if audio_shape_len == 1:
                self.audio = self.audio[:max_len]
            # stereo
            else:
                self.audio = self.audio[:, :max_len]
        # pad the signal with 0
        elif sig_len < max_len:
            # Length of padding to add to the beggining and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # pad with zeros
            # mono
            if audio_shape_len == 1:
                pad_begin = np.zeros(pad_begin_len)
                pad_end = np.zeros(pad_end_len)
                self.audio = np.concatenate((pad_begin, self.audio, pad_end))
            # stereo
            else:
                pad_begin = np.zeros((num_rows, pad_begin_len))
                pad_end = np.zeros((num_rows, pad_end_len))
                self.audio = np.concatenate(
                    (pad_begin, self.audio, pad_end), axis=1)

            self.length = librosa.get_duration(y=self.audio, sr=self.sr)

    def spectrogram(self, n_mels=64, n_ftt=1024, hop_len=None, top_db=80.0):
        """
        Generates the mel spectrogram. A Mel Spectrogram makes two important changes relative to a regular Spectrogram that plots Frequency vs Time:
          - It uses the Mel Scale instead of Frequency on the y-axis.
          - t uses the Decibel Scale instead of Amplitude to indicate colors.
        Deep learning models usually use this rather than a simple Spectrogram
        The Mel Spectrogram is then modified to use the Decibel Scale instead of Amplitude because most spectrograms are dark and do not carry enough useful information 
        Args:
          audio: ndarray
            Audio time series
          sr: number
            Sampling rate
          n_mels: int
            Number of mel bin
          n_ftt: int
            length of the FFT window
          hop_len: int
            number of samples between successive frames
          top_db: float
            threshold the output at top_db below the peak: max(20 * log10(S)) - top_db
        """
        # spec has shape [channel, n_mels, time] where channel is mono, stereo, etc
        audio_norm = librosa.util.normalize(self.audio)
        spec = librosa.feature.melspectrogram(
            y=audio_norm, sr=self.sr, n_fft=n_ftt, hop_length=hop_len, n_mels=n_mels)
        # convert to decibels
        spec = librosa.amplitude_to_db(S=spec, top_db=top_db)
        return(spec)
    
    def get_mfccs_generator(self, spectrogram, n_mfcc):
        mfccs = librosa.feature.mfcc(S=spectrogram, n_mfcc=n_mfcc)
        row = [[0, mfccs]]
        df = pd.DataFrame(row, columns=['file_num', 'mfccs'])

        predict_mfccs = np.array([df['mfccs'].iloc[i]
                                for i in range(len(df))], dtype='object')
        predict_mfccs = predict_mfccs.reshape(
            predict_mfccs.shape[0], predict_mfccs.shape[1], predict_mfccs.shape[2], 1)

        # generate predictor data, using dummy class
        predictor_gen = SingleInputGenerator(
            X1=predict_mfccs, Y=[1], batch_size=1, n_classes=2, shuffle=False)
        
        return predictor_gen
