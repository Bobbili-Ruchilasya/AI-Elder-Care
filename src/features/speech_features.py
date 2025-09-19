"""
Speech Feature Extraction Module
Extracts prosodic, acoustic, and emotional features from speech audio
"""

import librosa
import numpy as np
import torch
import torchaudio
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class SpeechFeatureExtractor:
    """
    Advanced speech feature extraction for loneliness detection
    Focuses on prosodic features, voice quality, and emotional indicators
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.feature_dim = 128  # Total feature dimension
        
    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive speech features from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing various feature types
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract different feature types
        features = {
            'prosodic': self._extract_prosodic_features(audio, sr),
            'spectral': self._extract_spectral_features(audio, sr),
            'temporal': self._extract_temporal_features(audio, sr),
            'emotional': self._extract_emotional_features(audio, sr),
            'voice_quality': self._extract_voice_quality_features(audio, sr)
        }
        
        # Combine all features
        combined_features = np.concatenate([
            features['prosodic'],
            features['spectral'], 
            features['temporal'],
            features['emotional'],
            features['voice_quality']
        ])
        
        features['combined'] = combined_features
        return features
    
    def _extract_prosodic_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract prosodic features (pitch, rhythm, stress patterns)"""
        features = []
        
        # Fundamental frequency (F0) features
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # F0 statistics
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features.extend([
                np.mean(f0_clean),           # Mean pitch
                np.std(f0_clean),            # Pitch variation
                np.max(f0_clean) - np.min(f0_clean),  # Pitch range
                np.percentile(f0_clean, 75) - np.percentile(f0_clean, 25),  # IQR
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Voice activity and pausing patterns
        rms = librosa.feature.rms(y=audio)[0]
        voice_activity = rms > np.percentile(rms, 20)
        
        # Pause duration features
        pause_segments = self._get_pause_segments(voice_activity)
        if len(pause_segments) > 0:
            features.extend([
                np.mean(pause_segments),     # Mean pause duration
                np.std(pause_segments),      # Pause variation
                len(pause_segments) / (len(audio) / sr)  # Pause frequency
            ])
        else:
            features.extend([0, 0, 0])
        
        # Speech rate
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        speech_rate = len(onset_frames) / (len(audio) / sr)
        features.append(speech_rate)
        
        return np.array(features)
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral features (MFCC, spectral centroid, etc.)"""
        features = []
        
        # MFCCs (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1)
        ])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])
        
        return np.concatenate(features).flatten()
    
    def _extract_temporal_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract temporal features (rhythm, timing patterns)"""
        features = []
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(tempo)
        
        # Rhythm regularity
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            features.extend([
                np.mean(beat_intervals),
                np.std(beat_intervals)
            ])
        else:
            features.extend([0, 0])
        
        # Energy envelope features
        rms = librosa.feature.rms(y=audio)[0]
        features.extend([
            np.mean(rms),
            np.std(rms),
            np.max(rms),
            np.min(rms)
        ])
        
        return np.array(features)
    
    def _extract_emotional_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract emotional indicators from speech"""
        features = []
        
        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend([
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        
        # Spectral contrast (energy distribution)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend([
            np.mean(contrast, axis=1),
            np.std(contrast, axis=1)
        ])
        
        # Tonnetz (harmonic network features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features.extend([
            np.mean(tonnetz, axis=1),
            np.std(tonnetz, axis=1)
        ])
        
        return np.concatenate(features).flatten()
    
    def _extract_voice_quality_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract voice quality indicators"""
        features = []
        
        # Jitter and shimmer approximation using pitch and amplitude variation
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 1:
            # Approximate jitter (pitch perturbation)
            jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0
            features.append(jitter)
        else:
            features.append(0)
        
        # Approximate shimmer (amplitude perturbation)
        rms = librosa.feature.rms(y=audio)[0]
        if len(rms) > 1:
            shimmer = np.std(np.diff(rms)) / np.mean(rms) if np.mean(rms) > 0 else 0
            features.append(shimmer)
        else:
            features.append(0)
        
        # Harmonics-to-noise ratio approximation
        hnr = self._estimate_hnr(audio, sr)
        features.append(hnr)
        
        return np.array(features)
    
    def _get_pause_segments(self, voice_activity: np.ndarray) -> List[float]:
        """Identify pause segments in speech"""
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, is_voice in enumerate(voice_activity):
            if not is_voice and not in_pause:
                # Start of pause
                in_pause = True
                pause_start = i
            elif is_voice and in_pause:
                # End of pause
                pause_duration = (i - pause_start) * (1 / self.sample_rate)
                pauses.append(pause_duration)
                in_pause = False
        
        return pauses
    
    def _estimate_hnr(self, audio: np.ndarray, sr: int) -> float:
        """Estimate harmonics-to-noise ratio"""
        # Simple HNR estimation using autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find the peak (fundamental period)
        if len(autocorr) > 1:
            peak_idx = np.argmax(autocorr[1:]) + 1
            hnr = 10 * np.log10(autocorr[peak_idx] / (autocorr[0] - autocorr[peak_idx] + 1e-10))
            return max(0, hnr)  # Clip negative values
        return 0

    def extract_features_batch(self, audio_paths: List[str]) -> np.ndarray:
        """Extract features from multiple audio files"""
        features_list = []
        for path in audio_paths:
            try:
                features = self.extract_features(path)
                features_list.append(features['combined'])
            except Exception as e:
                print(f"Error processing {path}: {e}")
                # Add zero features for failed extractions
                features_list.append(np.zeros(self.feature_dim))
        
        return np.array(features_list)
