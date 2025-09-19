"""
Advanced Visualization Components for Explainability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
import torch

class ExplainabilityDashboard:
    """
    Interactive dashboard for model explainability
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#7A7A7A'
        }
    
    def create_comprehensive_explanation_plot(self, explanations: Dict) -> go.Figure:
        """Create a comprehensive explanation dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Summary', 'Feature Importance', 
                          'Attention Weights', 'SHAP Values'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        prediction = explanations.get('prediction', {})
        
        # 1. Prediction Summary (Gauge)
        loneliness_score = prediction.get('loneliness_score', 0)
        confidence = prediction.get('confidence', 0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=loneliness_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Loneliness Score<br>Confidence: {confidence:.2f}"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "orange"},
                        {'range': [0.8, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Feature Importance
        if 'feature_importance' in explanations:
            importance = explanations['feature_importance']
            if 'modality_contributions' in importance:
                contrib = importance['modality_contributions']
                
                fig.add_trace(
                    go.Bar(
                        x=['Speech', 'Text'],
                        y=[contrib['speech'], contrib['text']],
                        marker_color=[self.color_palette['primary'], self.color_palette['secondary']],
                        name='Modality Contribution'
                    ),
                    row=1, col=2
                )
        
        # 3. Attention Weights (Text)
        if 'attention' in explanations and 'text_attention' in explanations['attention']:
            text_attn = explanations['attention']['text_attention']
            top_tokens = text_attn.get('top_tokens', [])[:5]
            
            if top_tokens:
                tokens, weights = zip(*top_tokens)
                fig.add_trace(
                    go.Bar(
                        x=list(tokens),
                        y=list(weights),
                        marker_color=self.color_palette['accent'],
                        name='Text Attention'
                    ),
                    row=2, col=1
                )
        
        # 4. SHAP Values
        if 'shap' in explanations and 'speech_shap' in explanations['shap']:
            shap_data = explanations['shap']['speech_shap']
            top_features = shap_data.get('top_features', [])[:5]
            
            if top_features:
                features, values = zip(*top_features)
                colors = ['red' if v < 0 else 'green' for v in values]
                
                fig.add_trace(
                    go.Bar(
                        x=list(features),
                        y=list(values),
                        marker_color=colors,
                        name='SHAP Values'
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Loneliness Detection - Model Explanation Dashboard",
            title_x=0.5
        )
        
        # Update axes titles
        fig.update_xaxes(title_text="Modality", row=1, col=2)
        fig.update_yaxes(title_text="Contribution", row=1, col=2)
        fig.update_xaxes(title_text="Tokens", row=2, col=1)
        fig.update_yaxes(title_text="Attention Weight", row=2, col=1)
        fig.update_xaxes(title_text="Features", row=2, col=2)
        fig.update_yaxes(title_text="SHAP Value", row=2, col=2)
        
        return fig
    
    def create_attention_heatmap(self, attention_data: Dict) -> go.Figure:
        """Create attention heatmap visualization"""
        
        if 'text_attention' not in attention_data:
            return go.Figure()
        
        text_attn = attention_data['text_attention']
        tokens = text_attn.get('tokens', [])
        weights = text_attn.get('weights', [])
        
        if not tokens or not weights:
            return go.Figure()
        
        # Create heatmap data
        heatmap_data = np.array(weights).reshape(1, -1)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=tokens,
            y=['Attention'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="Attention Weight")
        ))
        
        fig.update_layout(
            title='Text Attention Heatmap',
            xaxis_title='Tokens',
            yaxis_title='Layer',
            height=200,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_speech_feature_radar(self, speech_features: np.ndarray, 
                                   feature_names: List[str]) -> go.Figure:
        """Create radar chart for speech features"""
        
        # Normalize features for radar chart
        normalized_features = (speech_features - np.min(speech_features)) / (np.max(speech_features) - np.min(speech_features) + 1e-8)
        
        # Select top features for visualization
        top_indices = np.argsort(np.abs(speech_features))[-8:]
        selected_features = normalized_features[top_indices]
        selected_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in top_indices]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=selected_features,
            theta=selected_names,
            fill='toself',
            name='Speech Features',
            line_color=self.color_palette['primary']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Speech Feature Profile"
        )
        
        return fig
    
    def create_lime_explanation_plot(self, lime_data: Dict) -> go.Figure:
        """Create LIME explanation visualization"""
        
        if 'lime_explanation' not in lime_data:
            return go.Figure()
        
        lime_exp = lime_data['lime_explanation']
        words = lime_exp.get('words', [])
        scores = lime_exp.get('scores', [])
        
        if not words or not scores:
            return go.Figure()
        
        # Color code based on positive/negative influence
        colors = ['green' if score > 0 else 'red' for score in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                x=words,
                y=scores,
                marker_color=colors,
                text=[f'{score:.3f}' for score in scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='LIME Text Explanation',
            xaxis_title='Words',
            yaxis_title='Influence Score',
            xaxis_tickangle=-45,
            height=400
        )
        
        return fig
    
    def create_temporal_attention_plot(self, speech_attention: Dict) -> go.Figure:
        """Create temporal attention visualization for speech"""
        
        weights = speech_attention.get('weights', [])
        if not weights:
            return go.Figure()
        
        # Create time series
        time_steps = list(range(len(weights)))
        
        fig = go.Figure()
        
        # Add attention line
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=weights,
            mode='lines+markers',
            name='Attention Weight',
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=6)
        ))
        
        # Add shaded regions for high attention
        threshold = np.percentile(weights, 75)
        high_attention_regions = np.where(np.array(weights) > threshold)[0]
        
        if len(high_attention_regions) > 0:
            for region in high_attention_regions:
                fig.add_vrect(
                    x0=region-0.5, x1=region+0.5,
                    fillcolor=self.color_palette['accent'],
                    opacity=0.3,
                    layer="below", line_width=0
                )
        
        fig.update_layout(
            title='Temporal Attention Pattern in Speech',
            xaxis_title='Time Steps',
            yaxis_title='Attention Weight',
            height=400
        )
        
        return fig
    
    def create_confidence_meter(self, confidence_score: float) -> go.Figure:
        """Create confidence meter visualization"""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Confidence"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': self.color_palette['success']},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.8], 'color': "yellow"},
                    {'range': [0.8, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        
        return fig
    
    def create_feature_comparison_plot(self, explanations: Dict, baseline_features: Optional[Dict] = None) -> go.Figure:
        """Compare current features with baseline/normal features"""
        
        if not baseline_features:
            # Use neutral baseline
            baseline_features = {'speech': 0.5, 'text': 0.5}
        
        feature_importance = explanations.get('feature_importance', {})
        current_contributions = feature_importance.get('modality_contributions', {'speech': 0.5, 'text': 0.5})
        
        categories = ['Speech Features', 'Text Features']
        current_values = [current_contributions['speech'], current_contributions['text']]
        baseline_values = [baseline_features['speech'], baseline_features['text']]
        
        fig = go.Figure()
        
        # Current values
        fig.add_trace(go.Bar(
            name='Current',
            x=categories,
            y=current_values,
            marker_color=self.color_palette['primary']
        ))
        
        # Baseline values
        fig.add_trace(go.Bar(
            name='Baseline',
            x=categories,
            y=baseline_values,
            marker_color=self.color_palette['neutral']
        ))
        
        fig.update_layout(
            title='Feature Contribution Comparison',
            xaxis_title='Feature Type',
            yaxis_title='Contribution Score',
            barmode='group',
            height=400
        )
        
        return fig

class ReportGenerator:
    """
    Generate comprehensive explanation reports
    """
    
    def __init__(self):
        self.dashboard = ExplainabilityDashboard()
    
    def generate_explanation_report(self, explanations: Dict, 
                                   save_path: Optional[str] = None) -> str:
        """Generate comprehensive explanation report"""
        
        report = []
        
        # Header
        report.append("# Loneliness Detection - Explanation Report\n")
        report.append("---\n")
        
        # Prediction Summary
        prediction = explanations.get('prediction', {})
        score = prediction.get('loneliness_score', 0)
        confidence = prediction.get('confidence', 0)
        risk_level = prediction.get('risk_level', 'Unknown')
        
        report.append("## Prediction Summary\n")
        report.append(f"- **Loneliness Score**: {score:.3f}")
        report.append(f"- **Confidence**: {confidence:.3f}")
        report.append(f"- **Risk Level**: {risk_level}\n")
        
        # Natural Language Explanation
        if 'natural_language' in explanations:
            report.append("## Explanation\n")
            report.append(f"{explanations['natural_language']}\n")
        
        # Feature Analysis
        if 'feature_importance' in explanations:
            importance = explanations['feature_importance']
            report.append("## Feature Analysis\n")
            
            # Modality contributions
            if 'modality_contributions' in importance:
                contrib = importance['modality_contributions']
                report.append("### Modality Contributions")
                report.append(f"- **Speech**: {contrib['speech']:.1%}")
                report.append(f"- **Text**: {contrib['text']:.1%}\n")
            
            # Top speech features
            if 'top_speech_features' in importance:
                top_speech = importance['top_speech_features'][:5]
                report.append("### Top Speech Features")
                for feature, value in top_speech:
                    report.append(f"- {feature}: {value:.3f}")
                report.append("")
        
        # Attention Analysis
        if 'attention' in explanations:
            attention = explanations['attention']
            report.append("## Attention Analysis\n")
            
            # Text attention
            if 'text_attention' in attention:
                top_tokens = attention['text_attention'].get('top_tokens', [])[:5]
                if top_tokens:
                    report.append("### Most Attended Words")
                    for token, weight in top_tokens:
                        report.append(f"- {token}: {weight:.3f}")
                    report.append("")
        
        # SHAP Analysis
        if 'shap' in explanations:
            shap_data = explanations['shap']
            if 'speech_shap' in shap_data:
                top_shap = shap_data['speech_shap'].get('top_features', [])[:5]
                if top_shap:
                    report.append("## SHAP Feature Importance\n")
                    for feature, value in top_shap:
                        direction = "increases" if value > 0 else "decreases"
                        report.append(f"- {feature}: {direction} loneliness by {abs(value):.3f}")
                    report.append("")
        
        # Recommendations
        report.append("## Recommendations\n")
        if score > 0.7:
            report.append("- **High Risk**: Immediate intervention recommended")
            report.append("- Consider professional counseling or social support programs")
        elif score > 0.5:
            report.append("- **Moderate Risk**: Monitor closely and provide social activities")
            report.append("- Encourage family visits and community engagement")
        else:
            report.append("- **Low Risk**: Continue regular check-ins")
            report.append("- Maintain current social support systems")
        
        # Compile report
        full_report = "\n".join(report)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
        
        return full_report
