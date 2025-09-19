"""
Multimodal Feature Fusion Module
Combines speech and text features for joint analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from .speech_features import SpeechFeatureExtractor
from .text_features import TextFeatureExtractor

class MultimodalFeatureFusion:
    """
    Combines speech and text features using various fusion strategies
    """
    
    def __init__(self, fusion_strategy: str = 'attention'):
        """
        Initialize multimodal fusion
        
        Args:
            fusion_strategy: 'concatenation', 'attention', 'weighted', 'tensor'
        """
        self.speech_extractor = SpeechFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.fusion_strategy = fusion_strategy
        
        # Feature dimensions
        self.speech_dim = 128
        self.text_dim = 85
        self.fused_dim = self._get_fused_dim()
        
    def _get_fused_dim(self) -> int:
        """Get the dimension of fused features"""
        if self.fusion_strategy == 'concatenation':
            return self.speech_dim + self.text_dim
        elif self.fusion_strategy in ['attention', 'weighted']:
            return max(self.speech_dim, self.text_dim)
        elif self.fusion_strategy == 'tensor':
            return self.speech_dim * self.text_dim
        else:
            return self.speech_dim + self.text_dim
    
    def extract_multimodal_features(self, 
                                   audio_path: Optional[str] = None,
                                   text: Optional[str] = None,
                                   audio_features: Optional[np.ndarray] = None,
                                   text_features: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract and fuse multimodal features
        
        Args:
            audio_path: Path to audio file
            text: Text string
            audio_features: Pre-extracted audio features
            text_features: Pre-extracted text features
            
        Returns:
            Dictionary with individual and fused features
        """
        # Extract individual features if not provided
        if audio_features is None and audio_path is not None:
            speech_feats = self.speech_extractor.extract_features(audio_path)
            audio_features = speech_feats['combined']
        
        if text_features is None and text is not None:
            text_feats = self.text_extractor.extract_features(text)
            text_features = text_feats['combined']
        
        # Ensure we have both modalities
        if audio_features is None:
            audio_features = np.zeros(self.speech_dim)
        if text_features is None:
            text_features = np.zeros(self.text_dim)
        
        # Normalize features
        audio_features = self._normalize_features(audio_features)
        text_features = self._normalize_features(text_features)
        
        # Apply fusion strategy
        fused_features = self._fuse_features(audio_features, text_features)
        
        return {
            'speech': audio_features,
            'text': text_features,
            'fused': fused_features
        }
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance"""
        if np.std(features) > 0:
            return (features - np.mean(features)) / np.std(features)
        return features
    
    def _fuse_features(self, speech_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """Apply the selected fusion strategy"""
        
        if self.fusion_strategy == 'concatenation':
            return self._concatenation_fusion(speech_features, text_features)
        elif self.fusion_strategy == 'attention':
            return self._attention_fusion(speech_features, text_features)
        elif self.fusion_strategy == 'weighted':
            return self._weighted_fusion(speech_features, text_features)
        elif self.fusion_strategy == 'tensor':
            return self._tensor_fusion(speech_features, text_features)
        else:
            # Default to concatenation
            return self._concatenation_fusion(speech_features, text_features)
    
    def _concatenation_fusion(self, speech_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """Simple concatenation of features"""
        return np.concatenate([speech_features, text_features])
    
    def _attention_fusion(self, speech_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """Attention-based fusion (simplified version)"""
        # Pad features to same dimension
        max_dim = max(len(speech_features), len(text_features))
        
        speech_padded = np.pad(speech_features, (0, max_dim - len(speech_features)), 'constant')
        text_padded = np.pad(text_features, (0, max_dim - len(text_features)), 'constant')
        
        # Simple attention weights based on feature magnitude
        speech_weight = np.mean(np.abs(speech_features))
        text_weight = np.mean(np.abs(text_features))
        
        # Normalize weights
        total_weight = speech_weight + text_weight + 1e-8
        speech_weight /= total_weight
        text_weight /= total_weight
        
        # Weighted combination
        fused = speech_weight * speech_padded + text_weight * text_padded
        return fused
    
    def _weighted_fusion(self, speech_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """Weighted average fusion with learned weights"""
        # For simplicity, use fixed weights based on modality reliability
        speech_weight = 0.6  # Speech often contains more emotional information
        text_weight = 0.4    # Text provides semantic content
        
        # Pad to same dimension
        max_dim = max(len(speech_features), len(text_features))
        speech_padded = np.pad(speech_features, (0, max_dim - len(speech_features)), 'constant')
        text_padded = np.pad(text_features, (0, max_dim - len(text_features)), 'constant')
        
        return speech_weight * speech_padded + text_weight * text_padded
    
    def _tensor_fusion(self, speech_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """Tensor (outer product) fusion"""
        # Create outer product and flatten
        tensor_product = np.outer(speech_features, text_features)
        return tensor_product.flatten()

class CrossModalAttention(nn.Module):
    """
    Neural network module for cross-modal attention
    """
    
    def __init__(self, speech_dim: int, text_dim: int, hidden_dim: int = 64):
        super(CrossModalAttention, self).__init__()
        
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, speech_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modal attention
        
        Args:
            speech_features: (batch_size, speech_dim)
            text_features: (batch_size, text_dim)
            
        Returns:
            Fused features: (batch_size, hidden_dim)
        """
        # Project to common space
        speech_proj = self.speech_proj(speech_features).unsqueeze(1)  # (batch, 1, hidden)
        text_proj = self.text_proj(text_features).unsqueeze(1)        # (batch, 1, hidden)
        
        # Concatenate for attention
        combined = torch.cat([speech_proj, text_proj], dim=1)  # (batch, 2, hidden)
        
        # Apply self-attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Pool the attended features
        pooled = torch.mean(attended, dim=1)  # (batch, hidden)
        
        # Final projection
        output = self.output_proj(pooled)
        
        return output

class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns modality weights dynamically
    """
    
    def __init__(self, speech_dim: int, text_dim: int):
        super(AdaptiveFusion, self).__init__()
        
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        
        # Modality encoders
        self.speech_encoder = nn.Sequential(
            nn.Linear(speech_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, speech_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Adaptive fusion with learned weights
        
        Args:
            speech_features: (batch_size, speech_dim)
            text_features: (batch_size, text_dim)
            
        Returns:
            Fused features: (batch_size, 32)
        """
        # Encode modalities
        speech_encoded = self.speech_encoder(speech_features)
        text_encoded = self.text_encoder(text_features)
        
        # Concatenate for gate input
        gate_input = torch.cat([speech_encoded, text_encoded], dim=1)
        
        # Compute fusion weights
        weights = self.fusion_gate(gate_input)  # (batch, 2)
        speech_weight = weights[:, 0:1]  # (batch, 1)
        text_weight = weights[:, 1:2]    # (batch, 1)
        
        # Apply weights and fuse
        fused = speech_weight * speech_encoded + text_weight * text_encoded
        
        return fused
