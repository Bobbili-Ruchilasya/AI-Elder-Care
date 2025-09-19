"""
Ensemble Model Architecture for Loneliness Detection
Combines BERT for text analysis, CNN/RNN for speech, with cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional
import numpy as np

class TextBranch(nn.Module):
    """
    BERT-based text encoder for semantic understanding
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", hidden_dim: int = 256, freeze_bert: bool = False):
        super(TextBranch, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Additional layers for loneliness-specific features
        self.bert_hidden_size = self.bert.config.hidden_size
        
        self.text_processor = nn.Sequential(
            nn.Linear(self.bert_hidden_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention for important tokens
        self.token_attention = nn.MultiheadAttention(
            embed_dim=self.bert_hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through text branch
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Dictionary with text representations
        """
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get token-level representations
        token_embeddings = bert_output.last_hidden_state  # (batch, seq_len, hidden)
        
        # Apply token attention
        attended_tokens, attention_weights = self.token_attention(
            token_embeddings, token_embeddings, token_embeddings,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Pool representations
        pooled_output = bert_output.pooler_output  # (batch, hidden)
        
        # Additional processing
        text_features = self.text_processor(pooled_output)  # (batch, hidden_dim)
        
        return {
            'features': text_features,
            'pooled': pooled_output,
            'token_embeddings': attended_tokens,
            'attention_weights': attention_weights
        }

class SpeechBranch(nn.Module):
    """
    CNN-RNN hybrid for speech feature processing
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super(SpeechBranch, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1D CNN for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(32)  # Reduce to fixed length
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism for important temporal features
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, speech_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through speech branch
        
        Args:
            speech_features: Speech features (batch_size, feature_dim)
            
        Returns:
            Dictionary with speech representations
        """
        batch_size = speech_features.size(0)
        
        # Reshape for CNN (batch, 1, feature_dim)
        x = speech_features.unsqueeze(1)
        
        # Apply CNN layers
        conv_features = self.conv_layers(x)  # (batch, 128, 32)
        
        # Transpose for LSTM (batch, seq_len, features)
        conv_features = conv_features.transpose(1, 2)  # (batch, 32, 128)
        
        # Apply LSTM
        lstm_output, (hidden, cell) = self.lstm(conv_features)  # (batch, 32, hidden_dim)
        
        # Apply temporal attention
        attention_weights = self.temporal_attention(lstm_output)  # (batch, 32, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        attended_features = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden_dim)
        
        # Final projection
        speech_output = self.output_projection(attended_features)
        
        return {
            'features': speech_output,
            'lstm_output': lstm_output,
            'attention_weights': attention_weights.squeeze(-1),
            'conv_features': conv_features
        }

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention for fusing speech and text representations
    """
    
    def __init__(self, speech_dim: int, text_dim: int, fusion_dim: int = 256):
        super(CrossModalAttentionFusion, self).__init__()
        
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        
        # Project to common space
        self.speech_projection = nn.Linear(speech_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Self-attention for final representation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(fusion_dim)
        self.layer_norm2 = nn.LayerNorm(fusion_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
    
    def forward(self, speech_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Cross-modal attention fusion
        
        Args:
            speech_features: Speech representations (batch_size, speech_dim)
            text_features: Text representations (batch_size, text_dim)
            
        Returns:
            Fused representations
        """
        # Project to common space
        speech_proj = self.speech_projection(speech_features).unsqueeze(1)  # (batch, 1, fusion_dim)
        text_proj = self.text_projection(text_features).unsqueeze(1)        # (batch, 1, fusion_dim)
        
        # Cross-modal attention (speech attends to text and vice versa)
        speech_attended, speech_attn = self.cross_attention(
            speech_proj, text_proj, text_proj
        )
        text_attended, text_attn = self.cross_attention(
            text_proj, speech_proj, speech_proj
        )
        
        # Add residual connections and layer norm
        speech_attended = self.layer_norm1(speech_proj + speech_attended)
        text_attended = self.layer_norm1(text_proj + text_attended)
        
        # Concatenate and apply self-attention
        combined = torch.cat([speech_attended, text_attended], dim=1)  # (batch, 2, fusion_dim)
        
        fused, self_attn = self.self_attention(combined, combined, combined)
        
        # Layer norm and FFN
        fused = self.layer_norm2(combined + fused)
        fused_output = self.ffn(fused)
        
        # Pool to single representation
        pooled_output = torch.mean(fused_output, dim=1)  # (batch, fusion_dim)
        
        return {
            'fused_features': pooled_output,
            'speech_attention': speech_attn,
            'text_attention': text_attn,
            'self_attention': self_attn
        }

class LonelinessDetectionModel(nn.Module):
    """
    Complete ensemble model for loneliness detection
    """
    
    def __init__(self, 
                 speech_input_dim: int = 128,
                 text_model_name: str = "bert-base-uncased",
                 hidden_dim: int = 256,
                 num_classes: int = 1,  # Binary classification or regression
                 dropout: float = 0.3):
        super(LonelinessDetectionModel, self).__init__()
        
        self.speech_input_dim = speech_input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Individual branches
        self.text_branch = TextBranch(text_model_name, hidden_dim)
        self.speech_branch = SpeechBranch(speech_input_dim, hidden_dim)
        
        # Cross-modal fusion
        self.fusion = CrossModalAttentionFusion(hidden_dim, hidden_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                speech_features: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model
        
        Args:
            speech_features: Speech feature tensor (batch_size, speech_dim)
            input_ids: Text token IDs (batch_size, seq_len)
            attention_mask: Text attention mask (batch_size, seq_len)
            
        Returns:
            Dictionary with predictions and intermediate representations
        """
        # Process individual modalities
        text_output = self.text_branch(input_ids, attention_mask)
        speech_output = self.speech_branch(speech_features)
        
        # Cross-modal fusion
        fusion_output = self.fusion(
            speech_output['features'],
            text_output['features']
        )
        
        # Classification
        logits = self.classifier(fusion_output['fused_features'])
        
        # Apply appropriate activation for output
        if self.num_classes == 1:
            # Regression or binary classification
            predictions = torch.sigmoid(logits)
        else:
            # Multi-class classification
            predictions = F.softmax(logits, dim=1)
        
        return {
            'predictions': predictions,
            'logits': logits,
            'text_features': text_output['features'],
            'speech_features': speech_output['features'],
            'fused_features': fusion_output['fused_features'],
            'text_attention': text_output['attention_weights'],
            'speech_attention': speech_output['attention_weights'],
            'cross_modal_attention': {
                'speech_to_text': fusion_output['speech_attention'],
                'text_to_speech': fusion_output['text_attention']
            }
        }
    
    def predict_loneliness(self, 
                          speech_features: torch.Tensor,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> Dict[str, float]:
        """
        Predict loneliness score with confidence
        
        Args:
            speech_features: Speech features
            input_ids: Text token IDs
            attention_mask: Text attention mask
            
        Returns:
            Prediction results with confidence scores
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(speech_features, input_ids, attention_mask)
            
            loneliness_score = output['predictions'].squeeze().item()
            confidence = self._calculate_confidence(output)
            
            return {
                'loneliness_score': loneliness_score,
                'confidence': confidence,
                'risk_level': self._get_risk_level(loneliness_score),
                'contributing_factors': self._analyze_contributions(output)
            }
    
    def _calculate_confidence(self, output: Dict[str, torch.Tensor]) -> float:
        """Calculate prediction confidence based on attention patterns"""
        # Use attention weights to estimate confidence
        text_attn_entropy = -torch.sum(
            output['text_attention'] * torch.log(output['text_attention'] + 1e-8),
            dim=1
        ).mean().item()
        
        speech_attn_entropy = -torch.sum(
            output['speech_attention'] * torch.log(output['speech_attention'] + 1e-8),
            dim=1
        ).mean().item()
        
        # Lower entropy indicates more focused attention -> higher confidence
        avg_entropy = (text_attn_entropy + speech_attn_entropy) / 2
        confidence = 1.0 / (1.0 + avg_entropy)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        """Convert loneliness score to risk level"""
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _analyze_contributions(self, output: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze the contribution of each modality"""
        speech_contribution = torch.mean(torch.abs(output['speech_features'])).item()
        text_contribution = torch.mean(torch.abs(output['text_features'])).item()
        
        total = speech_contribution + text_contribution
        if total > 0:
            speech_ratio = speech_contribution / total
            text_ratio = text_contribution / total
        else:
            speech_ratio = text_ratio = 0.5
        
        return {
            'speech_contribution': speech_ratio,
            'text_contribution': text_ratio
        }
