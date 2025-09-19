"""
Gradio Alternative Interface for Loneliness Detection
Simpler interface option using Gradio
"""

import gradio as gr
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional
import tempfile
import os
from pathlib import Path

# Import our modules  
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble_model import LonelinessDetectionModel
from src.features import SpeechFeatureExtractor, TextFeatureExtractor
from src.explainability import ModelExplainer
from transformers import AutoTokenizer

class GradioLonelinessApp:
    """Gradio-based interface for loneliness detection"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.speech_extractor = SpeechFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.explainer = ModelExplainer(self.model, self.tokenizer, self.device)
    
    def _load_model(self):
        """Load the model (demo version)"""
        return LonelinessDetectionModel()
    
    def _generate_demo_speech_features(self, text: str) -> np.ndarray:
        """Generate synthetic speech features for demo"""
        base_features = np.random.randn(128) * 0.1
        
        # Adjust based on text sentiment
        lonely_words = ['alone', 'lonely', 'isolated', 'sad', 'empty']
        social_words = ['friends', 'family', 'together', 'happy', 'love']
        
        lonely_count = sum(1 for word in lonely_words if word in text.lower())
        social_count = sum(1 for word in social_words if word in text.lower())
        
        if lonely_count > social_count:
            base_features[1] *= 0.5  # Lower pitch variation
            base_features[7] *= 0.7  # Slower speech
        else:
            base_features[1] *= 1.3  # Higher pitch variation
            base_features[7] *= 1.2  # Faster speech
        
        return base_features
    
    def analyze_loneliness(self, 
                          audio_file: Optional[str], 
                          text_input: str) -> Tuple[str, str, str, str]:
        """
        Main analysis function for Gradio interface
        
        Returns:
            Tuple of (loneliness_score, risk_level, confidence, explanation)
        """
        
        if not text_input.strip():
            return "Error", "Please provide text input", "", ""
        
        try:
            # Extract speech features
            if audio_file and os.path.exists(audio_file):
                speech_features = self.speech_extractor.extract_features(audio_file)['combined']
            else:
                speech_features = self._generate_demo_speech_features(text_input)
            
            # Get explanations
            explanations = self.explainer.explain_prediction(
                speech_features, text_input, ['attention']
            )
            
            prediction = explanations['prediction']
            loneliness_score = prediction['loneliness_score']
            risk_level = prediction['risk_level']
            confidence = prediction['confidence']
            natural_explanation = explanations.get('natural_language', 'No explanation available')
            
            # Format outputs
            score_text = f"{loneliness_score:.3f} / 1.000"
            confidence_text = f"{confidence:.1%}"
            
            return score_text, risk_level, confidence_text, natural_explanation
            
        except Exception as e:
            return "Error", f"Analysis failed: {str(e)}", "", ""
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .output-text {
            font-size: 16px;
            font-weight: bold;
        }
        """
        
        with gr.Blocks(css=css, title="AI Elder Care - Loneliness Detection") as interface:
            
            # Header
            gr.Markdown(
                """
                # 🤗 AI Elder Care - Loneliness Detection System
                
                This AI system analyzes speech and text to detect signs of loneliness in elderly individuals.
                Provide either audio input, text input, or both for analysis.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎵 Audio Input")
                    audio_input = gr.Audio(
                        label="Upload audio file or record",
                        type="filepath"
                    )
                    
                    gr.Markdown("### 📝 Text Input")
                    text_input = gr.Textbox(
                        label="Enter what the person said",
                        placeholder="Example: I've been feeling quite alone lately...",
                        lines=4
                    )
                    
                    # Sample text examples
                    gr.Markdown("#### Quick Examples:")
                    sample_lonely = gr.Button("😔 Lonely Example")
                    sample_social = gr.Button("😊 Social Example")
                    sample_mixed = gr.Button("🤔 Mixed Example")
                    
                    analyze_btn = gr.Button("🔍 Analyze for Loneliness", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Analysis Results")
                    
                    loneliness_score = gr.Textbox(
                        label="Loneliness Score",
                        elem_classes=["output-text"]
                    )
                    
                    risk_level = gr.Textbox(
                        label="Risk Level",
                        elem_classes=["output-text"]
                    )
                    
                    confidence = gr.Textbox(
                        label="Confidence",
                        elem_classes=["output-text"]
                    )
                    
                    explanation = gr.Textbox(
                        label="AI Explanation",
                        lines=4
                    )
            
            # Sample text button actions
            sample_lonely.click(
                lambda: "I feel so alone these days. Nobody calls me anymore, and I spend most of my time by myself.",
                outputs=text_input
            )
            
            sample_social.click(
                lambda: "I had a wonderful visit from my grandchildren today. We played games and had such a lovely time together.",
                outputs=text_input
            )
            
            sample_mixed.click(
                lambda: "Some days are better than others. I try to stay positive, but sometimes the loneliness creeps in.",
                outputs=text_input
            )
            
            # Main analysis action
            analyze_btn.click(
                fn=self.analyze_loneliness,
                inputs=[audio_input, text_input],
                outputs=[loneliness_score, risk_level, confidence, explanation]
            )
            
            # Information section
            gr.Markdown(
                """
                ---
                ### ℹ️ Understanding the Results
                
                **Loneliness Score**: 0.0 (not lonely) to 1.0 (very lonely)
                - 0.0-0.3: Low risk
                - 0.3-0.6: Moderate risk  
                - 0.6-0.8: High risk
                - 0.8-1.0: Very high risk
                
                **Important**: This is a supportive tool, not a medical diagnosis. 
                Consult healthcare professionals for proper assessment.
                """
            )
        
        return interface
    
    def launch(self, share=False, debug=False):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        interface.launch(share=share, debug=debug)

# Command line interface
if __name__ == "__main__":
    app = GradioLonelinessApp()
    app.launch(share=True, debug=True)
