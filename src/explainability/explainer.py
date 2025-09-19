"""
Explainability Framework for Loneliness Detection
Implements SHAP, LIME, and attention visualization for model interpretability
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Explainability libraries
import shap
from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance

# Text processing
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize

class ModelExplainer:
    """
    Comprehensive explainability framework for loneliness detection model
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize explainability tools
        self.lime_explainer = LimeTextExplainer(class_names=['Not Lonely', 'Lonely'])
        
        # Captum attributions
        self.integrated_gradients = IntegratedGradients(self.model)
        
        # SHAP explainer (will be initialized when needed)
        self.shap_explainer = None
        
        # Feature names for different modalities
        self.speech_feature_names = self._get_speech_feature_names()
        self.text_feature_categories = self._get_text_feature_categories()
    
    def _get_speech_feature_names(self) -> List[str]:
        """Get names for speech features"""
        return [
            # Prosodic features
            'mean_pitch', 'pitch_variation', 'pitch_range', 'pitch_iqr',
            'mean_pause_duration', 'pause_variation', 'pause_frequency', 'speech_rate',
            
            # Spectral features  
            'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5',
            'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3', 'mfcc_std_4', 'mfcc_std_5',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'zero_crossing_rate_mean', 'zero_crossing_rate_std',
            
            # Temporal features
            'tempo', 'beat_interval_mean', 'beat_interval_std',
            'rms_mean', 'rms_std', 'rms_max', 'rms_min',
            
            # Emotional features
            'chroma_mean_1', 'chroma_mean_2', 'chroma_std_1', 'chroma_std_2',
            'contrast_mean_1', 'contrast_mean_2', 'contrast_std_1', 'contrast_std_2',
            'tonnetz_mean_1', 'tonnetz_mean_2', 'tonnetz_std_1', 'tonnetz_std_2',
            
            # Voice quality
            'jitter', 'shimmer', 'hnr'
        ]
    
    def _get_text_feature_categories(self) -> Dict[str, List[str]]:
        """Get categories for text features"""
        return {
            'sentiment': ['positive_sentiment', 'neutral_sentiment', 'negative_sentiment', 
                         'compound_sentiment', 'polarity', 'subjectivity'],
            'linguistic': ['word_count', 'sentence_count', 'avg_words_per_sentence', 
                          'long_words', 'lexical_diversity'],
            'semantic': ['isolation_keywords', 'sadness_keywords', 'social_withdrawal_keywords',
                        'negative_emotion_keywords', 'social_connection_keywords', 'positive_emotion_keywords'],
            'emotional': ['anxiety_words', 'anger_words', 'joy_words', 'trust_words',
                         'surprise_words', 'disgust_words', 'anticipation_words']
        }
    
    def explain_prediction(self, 
                          speech_features: np.ndarray,
                          text: str,
                          explanation_types: List[str] = ['attention', 'shap', 'lime']) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for a prediction
        
        Args:
            speech_features: Speech feature vector
            text: Input text
            explanation_types: Types of explanations to generate
            
        Returns:
            Dictionary containing various explanations
        """
        explanations = {}
        
        # Prepare inputs
        speech_tensor = torch.FloatTensor(speech_features).unsqueeze(0).to(self.device)
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=512, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get model prediction and intermediate outputs
        self.model.eval()
        with torch.no_grad():
            output = self.model(speech_tensor, input_ids, attention_mask)
        
        # Base prediction info
        explanations['prediction'] = {
            'loneliness_score': output['predictions'].squeeze().item(),
            'confidence': self._calculate_confidence(output),
            'risk_level': self._get_risk_level(output['predictions'].squeeze().item())
        }
        
        # Attention-based explanations
        if 'attention' in explanation_types:
            explanations['attention'] = self._explain_attention(output, text, speech_features)
        
        # SHAP explanations
        if 'shap' in explanation_types:
            explanations['shap'] = self._explain_shap(speech_features, text)
        
        # LIME explanations
        if 'lime' in explanation_types:
            explanations['lime'] = self._explain_lime(text)
        
        # Feature importance
        explanations['feature_importance'] = self._analyze_feature_importance(output, speech_features, text)
        
        # Natural language explanation
        explanations['natural_language'] = self._generate_natural_explanation(explanations)
        
        return explanations
    
    def _explain_attention(self, model_output: Dict, text: str, speech_features: np.ndarray) -> Dict[str, Any]:
        """Generate attention-based explanations"""
        explanations = {}
        
        # Text attention weights
        text_attention = model_output['text_attention'].squeeze().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(text, return_tensors='pt')['input_ids'].squeeze()
        )
        
        # Filter out special tokens and get meaningful attention
        meaningful_tokens = []
        meaningful_attention = []
        for token, attn in zip(tokens, text_attention):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                meaningful_tokens.append(token.replace('##', ''))
                meaningful_attention.append(attn)
        
        explanations['text_attention'] = {
            'tokens': meaningful_tokens,
            'weights': meaningful_attention,
            'top_tokens': self._get_top_attention_tokens(meaningful_tokens, meaningful_attention, top_k=5)
        }
        
        # Speech attention weights
        speech_attention = model_output['speech_attention'].squeeze().cpu().numpy()
        explanations['speech_attention'] = {
            'weights': speech_attention.tolist(),
            'top_features': self._get_top_speech_features(speech_features, speech_attention)
        }
        
        # Cross-modal attention
        cross_modal = model_output['cross_modal_attention']
        explanations['cross_modal'] = {
            'speech_to_text': cross_modal['speech_to_text'].squeeze().cpu().numpy().tolist(),
            'text_to_speech': cross_modal['text_to_speech'].squeeze().cpu().numpy().tolist()
        }
        
        return explanations
    
    def _explain_shap(self, speech_features: np.ndarray, text: str) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        explanations = {}
        
        try:
            # Create a wrapper function for SHAP
            def model_wrapper(inputs):
                # inputs should be [speech_features, text] pairs
                predictions = []
                for speech_feat, txt in inputs:
                    # Prepare tensors
                    speech_tensor = torch.FloatTensor(speech_feat).unsqueeze(0).to(self.device)
                    encoding = self.tokenizer(
                        txt, truncation=True, padding='max_length',
                        max_length=512, return_tensors='pt'
                    )
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    
                    # Get prediction
                    with torch.no_grad():
                        output = self.model(speech_tensor, input_ids, attention_mask)
                        predictions.append(output['predictions'].squeeze().cpu().numpy())
                
                return np.array(predictions)
            
            # For speech features
            speech_shap_values = self._compute_speech_shap(speech_features, text)
            explanations['speech_shap'] = {
                'values': speech_shap_values.tolist(),
                'feature_names': self.speech_feature_names[:len(speech_shap_values)],
                'top_features': self._get_top_shap_features(speech_shap_values, self.speech_feature_names)
            }
            
        except Exception as e:
            explanations['error'] = f"SHAP computation failed: {str(e)}"
            explanations['speech_shap'] = {'values': [], 'feature_names': [], 'top_features': []}
        
        return explanations
    
    def _explain_lime(self, text: str) -> Dict[str, Any]:
        """Generate LIME explanations for text"""
        explanations = {}
        
        try:
            # Create prediction function for LIME
            def predict_fn(texts):
                predictions = []
                for txt in texts:
                    # Use zero speech features for text-only explanation
                    speech_tensor = torch.zeros(1, 128).to(self.device)
                    encoding = self.tokenizer(
                        txt, truncation=True, padding='max_length',
                        max_length=512, return_tensors='pt'
                    )
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(speech_tensor, input_ids, attention_mask)
                        pred = output['predictions'].squeeze().cpu().numpy()
                        predictions.append([1-pred, pred])  # [not_lonely, lonely]
                
                return np.array(predictions)
            
            # Generate LIME explanation
            lime_explanation = self.lime_explainer.explain_instance(
                text, predict_fn, num_features=10
            )
            
            # Extract explanation data
            explanations['lime_explanation'] = {
                'words': [item[0] for item in lime_explanation.as_list()],
                'scores': [item[1] for item in lime_explanation.as_list()],
                'prediction_probability': lime_explanation.predict_proba[1]
            }
            
        except Exception as e:
            explanations['error'] = f"LIME computation failed: {str(e)}"
            explanations['lime_explanation'] = {'words': [], 'scores': [], 'prediction_probability': 0.5}
        
        return explanations
    
    def _compute_speech_shap(self, speech_features: np.ndarray, text: str) -> np.ndarray:
        """Compute SHAP values for speech features"""
        # Create background dataset (use zeros as baseline)
        background = np.zeros((1, len(speech_features)))
        
        # Define prediction function
        def predict_speech(speech_batch):
            predictions = []
            for speech_feat in speech_batch:
                speech_tensor = torch.FloatTensor(speech_feat).unsqueeze(0).to(self.device)
                encoding = self.tokenizer(
                    text, truncation=True, padding='max_length',
                    max_length=512, return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    output = self.model(speech_tensor, input_ids, attention_mask)
                    predictions.append(output['predictions'].squeeze().cpu().numpy())
            
            return np.array(predictions)
        
        # Initialize SHAP explainer
        explainer = shap.Explainer(predict_speech, background)
        shap_values = explainer(speech_features.reshape(1, -1))
        
        return shap_values.values[0]
    
    def _analyze_feature_importance(self, model_output: Dict, speech_features: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze overall feature importance"""
        importance = {}
        
        # Modality contributions
        speech_contribution = torch.mean(torch.abs(model_output['speech_features'])).item()
        text_contribution = torch.mean(torch.abs(model_output['text_features'])).item()
        
        total_contribution = speech_contribution + text_contribution
        if total_contribution > 0:
            importance['modality_contributions'] = {
                'speech': speech_contribution / total_contribution,
                'text': text_contribution / total_contribution
            }
        else:
            importance['modality_contributions'] = {'speech': 0.5, 'text': 0.5}
        
        # Top contributing features by category
        importance['top_speech_features'] = self._get_top_speech_features(speech_features)
        importance['top_text_patterns'] = self._analyze_text_patterns(text)
        
        return importance
    
    def _get_top_attention_tokens(self, tokens: List[str], weights: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top tokens by attention weight"""
        token_weights = list(zip(tokens, weights))
        token_weights.sort(key=lambda x: x[1], reverse=True)
        return token_weights[:top_k]
    
    def _get_top_speech_features(self, speech_features: np.ndarray, attention_weights: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """Get top speech features by importance"""
        if attention_weights is not None:
            # Use attention weights
            weights = attention_weights
        else:
            # Use absolute feature values
            weights = np.abs(speech_features)
        
        feature_importance = []
        for i, (feature_name, weight) in enumerate(zip(self.speech_feature_names, weights)):
            if i < len(speech_features):
                feature_importance.append((feature_name, float(weight)))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        return feature_importance[:10]
    
    def _get_top_shap_features(self, shap_values: np.ndarray, feature_names: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top features by SHAP values"""
        feature_shap = []
        for i, (name, value) in enumerate(zip(feature_names, shap_values)):
            if i < len(shap_values):
                feature_shap.append((name, float(value)))
        
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_shap[:top_k]
    
    def _analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze important text patterns"""
        patterns = {}
        
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Loneliness-related keywords
        loneliness_keywords = {
            'isolation': ['alone', 'lonely', 'isolated', 'solitary'],
            'sadness': ['sad', 'depressed', 'down', 'blue'],
            'social': ['friends', 'family', 'people', 'visit']
        }
        
        for category, keywords in loneliness_keywords.items():
            found_keywords = [word for word in tokens if word in keywords]
            patterns[f'{category}_keywords'] = found_keywords
        
        # Sentiment indicators
        negative_words = ['not', 'no', 'never', 'nothing', 'nobody']
        patterns['negative_indicators'] = [word for word in tokens if word in negative_words]
        
        return patterns
    
    def _calculate_confidence(self, model_output: Dict) -> float:
        """Calculate prediction confidence"""
        # Use attention entropy as confidence measure
        text_attn = model_output['text_attention']
        speech_attn = model_output['speech_attention']
        
        # Calculate entropy (lower entropy = higher confidence)
        text_entropy = -torch.sum(text_attn * torch.log(text_attn + 1e-8)).item()
        speech_entropy = -torch.sum(speech_attn * torch.log(speech_attn + 1e-8)).item()
        
        avg_entropy = (text_entropy + speech_entropy) / 2
        confidence = 1.0 / (1.0 + avg_entropy / 10)  # Normalize
        
        return min(max(confidence, 0.0), 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        """Convert loneliness score to risk level"""
        if score < 0.3:
            return "Low Risk"
        elif score < 0.6:
            return "Moderate Risk"
        elif score < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _generate_natural_explanation(self, explanations: Dict) -> str:
        """Generate natural language explanation"""
        prediction = explanations['prediction']
        score = prediction['loneliness_score']
        risk_level = prediction['risk_level']
        
        explanation = f"The model predicts a loneliness score of {score:.2f}, indicating {risk_level.lower()}. "
        
        # Add attention-based insights
        if 'attention' in explanations:
            top_tokens = explanations['attention']['text_attention']['top_tokens']
            if top_tokens:
                important_words = [token for token, _ in top_tokens[:3]]
                explanation += f"Key words that influenced this prediction include: {', '.join(important_words)}. "
        
        # Add modality contributions
        if 'feature_importance' in explanations:
            contributions = explanations['feature_importance']['modality_contributions']
            if contributions['speech'] > contributions['text']:
                explanation += "The speech patterns were more indicative of loneliness than the text content. "
            else:
                explanation += "The text content was more indicative of loneliness than the speech patterns. "
        
        return explanation

class VisualizationEngine:
    """
    Visualization engine for explainability results
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def create_attention_visualization(self, explanations: Dict) -> Dict[str, Any]:
        """Create attention weight visualizations"""
        visualizations = {}
        
        if 'attention' in explanations:
            attention_data = explanations['attention']
            
            # Text attention heatmap
            if 'text_attention' in attention_data:
                visualizations['text_attention'] = self._create_text_attention_plot(attention_data['text_attention'])
            
            # Speech attention plot
            if 'speech_attention' in attention_data:
                visualizations['speech_attention'] = self._create_speech_attention_plot(attention_data['speech_attention'])
        
        return visualizations
    
    def _create_text_attention_plot(self, text_attention: Dict) -> go.Figure:
        """Create text attention visualization"""
        tokens = text_attention['tokens']
        weights = text_attention['weights']
        
        fig = go.Figure(data=[
            go.Bar(
                x=tokens,
                y=weights,
                text=[f'{w:.3f}' for w in weights],
                textposition='auto',
                marker_color=weights,
                colorscale='RdYlBu_r'
            )
        ])
        
        fig.update_layout(
            title='Text Attention Weights',
            xaxis_title='Tokens',
            yaxis_title='Attention Weight',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_speech_attention_plot(self, speech_attention: Dict) -> go.Figure:
        """Create speech attention visualization"""
        weights = speech_attention['weights']
        
        fig = go.Figure(data=[
            go.Scatter(
                y=weights,
                mode='lines+markers',
                name='Attention Weights',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            )
        ])
        
        fig.update_layout(
            title='Speech Feature Attention Weights',
            xaxis_title='Temporal Position',
            yaxis_title='Attention Weight'
        )
        
        return fig
    
    def create_feature_importance_plot(self, explanations: Dict) -> go.Figure:
        """Create feature importance visualization"""
        if 'feature_importance' not in explanations:
            return go.Figure()
        
        importance_data = explanations['feature_importance']
        
        # Modality contributions pie chart
        contributions = importance_data['modality_contributions']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Speech', 'Text'],
                values=[contributions['speech'], contributions['text']],
                hole=0.3,
                marker_colors=['lightblue', 'lightcoral']
            )
        ])
        
        fig.update_layout(
            title='Modality Contributions to Prediction',
            annotations=[dict(text='Contributions', x=0.5, y=0.5, font_size=12, showarrow=False)]
        )
        
        return fig
