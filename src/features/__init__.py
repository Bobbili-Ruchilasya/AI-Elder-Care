"""
Feature extraction package initialization
"""

from .speech_features import SpeechFeatureExtractor
from .text_features import TextFeatureExtractor
from .multimodal_fusion import MultimodalFeatureFusion, CrossModalAttention, AdaptiveFusion

__all__ = [
    'SpeechFeatureExtractor',
    'TextFeatureExtractor', 
    'MultimodalFeatureFusion',
    'CrossModalAttention',
    'AdaptiveFusion'
]
