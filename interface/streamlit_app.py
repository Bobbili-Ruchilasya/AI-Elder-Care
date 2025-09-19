"""
Streamlit Web Interface for Loneliness Detection
Real-time multimodal analysis with explanations
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble_model import LonelinessDetectionModel
from src.features import SpeechFeatureExtractor, TextFeatureExtractor
from src.explainability import ModelExplainer, ExplainabilityDashboard
from src.utils import Config
from transformers import AutoTokenizer

# Configure page
st.set_page_config(
    page_title="AI Elder Care - Loneliness Detection",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .moderate-risk {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class LonelinessDetectionApp:
    """Main application class for the Streamlit interface"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.explainer = None
        self.speech_extractor = SpeechFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.dashboard = ExplainabilityDashboard()
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
    
    def load_model(self):
        """Load the trained model"""
        try:
            # For demo purposes, create a model with random weights
            # In production, you would load a trained model
            self.model = LonelinessDetectionModel()
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.explainer = ModelExplainer(self.model, self.tokenizer, self.device)
            
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.session_state.model_loaded = False
    
    def run(self):
        """Main application runner"""
        
        # Header
        st.markdown('<h1 class="main-header">🤗 AI Elder Care - Loneliness Detection System</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        This AI system analyzes speech patterns and text content to detect signs of loneliness in elderly individuals, 
        providing early intervention opportunities and supporting mental health care.
        """)
        
        # Sidebar
        self.create_sidebar()
        
        # Load model if not already loaded
        if not st.session_state.model_loaded:
            if st.button("🚀 Load AI Model", type="primary"):
                self.load_model()
        
        if st.session_state.model_loaded:
            # Main interface tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎤 Real-time Analysis", 
                "📊 Analysis Dashboard", 
                "📈 History & Trends",
                "ℹ️ About & Help"
            ])
            
            with tab1:
                self.real_time_analysis_tab()
            
            with tab2:
                self.dashboard_tab()
            
            with tab3:
                self.history_tab()
                
            with tab4:
                self.about_tab()
        else:
            st.info("👆 Please load the AI model to begin analysis")
    
    def create_sidebar(self):
        """Create sidebar with settings and information"""
        
        st.sidebar.markdown("## ⚙️ Settings")
        
        # Analysis settings
        st.sidebar.markdown("### Analysis Parameters")
        threshold = st.sidebar.slider(
            "Loneliness Detection Threshold", 
            min_value=0.1, max_value=0.9, value=0.5, step=0.1,
            help="Threshold for classifying as lonely vs not lonely"
        )
        
        confidence_threshold = st.sidebar.slider(
            "Minimum Confidence", 
            min_value=0.5, max_value=0.95, value=0.7, step=0.05,
            help="Minimum confidence level for reliable predictions"
        )
        
        # Explanation settings
        st.sidebar.markdown("### Explanation Options")
        show_attention = st.sidebar.checkbox("Show Attention Weights", value=True)
        show_shap = st.sidebar.checkbox("Show SHAP Analysis", value=True)
        show_lime = st.sidebar.checkbox("Show LIME Explanation", value=False)
        
        # Model info
        st.sidebar.markdown("### 🤖 Model Information")
        if st.session_state.model_loaded:
            st.sidebar.success("✅ Model Status: Loaded")
            st.sidebar.info(f"🖥️ Device: {self.device.upper()}")
        else:
            st.sidebar.warning("⏳ Model Status: Not Loaded")
        
        # Store settings in session state
        st.session_state.threshold = threshold
        st.session_state.confidence_threshold = confidence_threshold
        st.session_state.show_attention = show_attention
        st.session_state.show_shap = show_shap
        st.session_state.show_lime = show_lime
    
    def real_time_analysis_tab(self):
        """Real-time analysis interface"""
        
        st.markdown("## 🎤 Real-time Loneliness Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎵 Audio Input")
            
            # Audio input options
            audio_option = st.radio(
                "Choose audio input method:",
                ["Record Audio", "Upload Audio File", "Use Sample Audio"]
            )
            
            audio_data = None
            audio_file_path = None
            
            if audio_option == "Record Audio":
                st.info("🎙️ Audio recording feature coming soon!")
                st.markdown("*Note: Use 'Upload Audio File' or 'Use Sample Audio' for now*")
            
            elif audio_option == "Upload Audio File":
                uploaded_file = st.file_uploader(
                    "Upload audio file (.wav, .mp3, .m4a)", 
                    type=['wav', 'mp3', 'm4a']
                )
                
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        audio_file_path = tmp_file.name
                    
                    st.audio(uploaded_file)
                    st.success("✅ Audio file uploaded successfully!")
            
            elif audio_option == "Use Sample Audio":
                st.info("🎵 Using sample audio for demonstration")
                audio_file_path = "sample_audio.wav"  # Placeholder
        
        with col2:
            st.markdown("### 📝 Text Input")
            
            text_input = st.text_area(
                "Enter what the person said (or transcript):",
                placeholder="Example: I've been feeling quite alone lately. My children don't visit as often as they used to, and the house feels so quiet...",
                height=150
            )
            
            # Sample text options
            sample_texts = {
                "Lonely Example": "I feel so alone these days. Nobody calls me anymore, and I spend most of my time by myself. The house is so quiet and empty.",
                "Social Example": "I had a wonderful visit from my grandchildren today. We played games and had such a lovely time together. I feel very blessed.",
                "Mixed Example": "Some days are better than others. I try to stay positive, but sometimes the loneliness creeps in when it gets dark."
            }
            
            selected_sample = st.selectbox("Or choose a sample text:", [""] + list(sample_texts.keys()))
            
            if selected_sample:
                text_input = sample_texts[selected_sample]
                st.rerun()
        
        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analyze for Loneliness", 
                type="primary", 
                use_container_width=True
            )
        
        # Perform analysis
        if analyze_button and text_input:
            with st.spinner("🧠 AI is analyzing the input..."):
                results = self.perform_analysis(audio_file_path, text_input)
                
                if results:
                    self.display_results(results)
                    
                    # Add to history
                    st.session_state.prediction_history.append({
                        'timestamp': pd.Timestamp.now(),
                        'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                        'loneliness_score': results['prediction']['loneliness_score'],
                        'risk_level': results['prediction']['risk_level'],
                        'confidence': results['prediction']['confidence']
                    })
        
        elif analyze_button and not text_input:
            st.warning("⚠️ Please provide text input for analysis")
    
    def perform_analysis(self, audio_file_path: Optional[str], text: str) -> Dict[str, Any]:
        """Perform loneliness analysis on the inputs"""
        try:
            # Extract speech features (use synthetic if no audio)
            if audio_file_path and os.path.exists(audio_file_path):
                speech_features = self.speech_extractor.extract_features(audio_file_path)['combined']
            else:
                # Generate synthetic speech features for demo
                speech_features = self._generate_demo_speech_features(text)
            
            # Get explanation settings
            explanation_types = []
            if st.session_state.show_attention:
                explanation_types.append('attention')
            if st.session_state.show_shap:
                explanation_types.append('shap')
            if st.session_state.show_lime:
                explanation_types.append('lime')
            
            # Generate explanations
            explanations = self.explainer.explain_prediction(
                speech_features, text, explanation_types
            )
            
            return explanations
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None
    
    def _generate_demo_speech_features(self, text: str) -> np.ndarray:
        """Generate synthetic speech features for demo purposes"""
        # Create features that correlate with text sentiment
        base_features = np.random.randn(128) * 0.1
        
        # Adjust based on text content (simple heuristic)
        lonely_words = ['alone', 'lonely', 'isolated', 'sad', 'empty', 'quiet']
        social_words = ['friends', 'family', 'together', 'visit', 'happy', 'love']
        
        lonely_count = sum(1 for word in lonely_words if word in text.lower())
        social_count = sum(1 for word in social_words if word in text.lower())
        
        if lonely_count > social_count:
            # Adjust features to indicate loneliness
            base_features[1] *= 0.5  # Lower pitch variation
            base_features[7] *= 0.7  # Slower speech
            base_features[42] -= 0.3  # Lower energy
        else:
            # Adjust features to indicate social connection
            base_features[1] *= 1.3  # Higher pitch variation
            base_features[7] *= 1.2  # Faster speech  
            base_features[42] += 0.2  # Higher energy
        
        return base_features
    
    def display_results(self, results: Dict[str, Any]):
        """Display analysis results"""
        
        prediction = results['prediction']
        loneliness_score = prediction['loneliness_score']
        confidence = prediction['confidence']
        risk_level = prediction['risk_level']
        
        # Main results
        st.markdown("## 📊 Analysis Results")
        
        # Risk level banner
        risk_class = "high-risk" if loneliness_score > 0.7 else "moderate-risk" if loneliness_score > 0.4 else "low-risk"
        
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h3>Risk Assessment: {risk_level}</h3>
            <h2>Loneliness Score: {loneliness_score:.2f}/1.00</h2>
            <p>Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Loneliness Score", 
                f"{loneliness_score:.2f}",
                delta=f"{loneliness_score - 0.5:.2f}" if loneliness_score != 0.5 else None
            )
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.metric("Risk Level", risk_level)
        
        with col4:
            # Determine if intervention is needed
            intervention = "Recommended" if loneliness_score > 0.6 else "Monitor" if loneliness_score > 0.3 else "None"
            st.metric("Intervention", intervention)
        
        # Visualizations
        if st.session_state.show_attention and 'attention' in results:
            self.display_attention_analysis(results['attention'])
        
        # Natural language explanation
        if 'natural_language' in results:
            st.markdown("### 🗣️ AI Explanation")
            st.info(results['natural_language'])
        
        # Feature contributions
        if 'feature_importance' in results:
            self.display_feature_importance(results['feature_importance'])
    
    def display_attention_analysis(self, attention_data: Dict):
        """Display attention analysis visualizations"""
        
        st.markdown("### 🔍 Attention Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'text_attention' in attention_data:
                text_attn = attention_data['text_attention']
                top_tokens = text_attn.get('top_tokens', [])
                
                if top_tokens:
                    st.markdown("#### 📝 Most Important Words")
                    
                    # Create attention visualization
                    tokens, weights = zip(*top_tokens[:8])
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(tokens),
                            y=list(weights),
                            marker_color=px.colors.sequential.Viridis,
                            text=[f'{w:.3f}' for w in weights],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Word Importance (Attention Weights)",
                        xaxis_title="Words",
                        yaxis_title="Attention Weight",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'speech_attention' in attention_data:
                st.markdown("#### 🎵 Speech Pattern Focus")
                
                weights = attention_data['speech_attention']['weights']
                
                if weights:
                    fig = go.Figure(data=[
                        go.Scatter(
                            y=weights,
                            mode='lines+markers',
                            name='Attention Weight',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4)
                        )
                    ])
                    
                    fig.update_layout(
                        title="Speech Feature Attention",
                        xaxis_title="Feature Index",
                        yaxis_title="Attention Weight",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def display_feature_importance(self, importance_data: Dict):
        """Display feature importance analysis"""
        
        st.markdown("### 📈 Feature Contribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Modality contributions
            if 'modality_contributions' in importance_data:
                contrib = importance_data['modality_contributions']
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Speech Analysis', 'Text Analysis'],
                        values=[contrib['speech'], contrib['text']],
                        hole=0.3,
                        marker_colors=['lightblue', 'lightcoral']
                    )
                ])
                
                fig.update_layout(
                    title="Analysis Contribution by Modality",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top speech features
            if 'top_speech_features' in importance_data:
                top_features = importance_data['top_speech_features'][:6]
                
                if top_features:
                    features, values = zip(*top_features)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(values),
                            y=list(features),
                            orientation='h',
                            marker_color='lightgreen'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Top Contributing Speech Features",
                        xaxis_title="Importance Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def dashboard_tab(self):
        """Analysis dashboard tab"""
        st.markdown("## 📊 Analysis Dashboard")
        
        if not st.session_state.prediction_history:
            st.info("No analysis history available. Perform some analyses first!")
            return
        
        # Create dashboard visualizations
        df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loneliness score distribution
            fig = px.histogram(
                df, x='loneliness_score', 
                title='Loneliness Score Distribution',
                nbins=20,
                color_discrete_sequence=['skyblue']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk level distribution
            risk_counts = df['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Risk Level Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline
        if len(df) > 1:
            fig = px.line(
                df, x='timestamp', y='loneliness_score',
                title='Loneliness Score Over Time',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def history_tab(self):
        """Analysis history tab"""
        st.markdown("## 📈 Analysis History & Trends")
        
        if not st.session_state.prediction_history:
            st.info("No analysis history available yet.")
            return
        
        df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(df))
        
        with col2:
            avg_score = df['loneliness_score'].mean()
            st.metric("Average Score", f"{avg_score:.2f}")
        
        with col3:
            high_risk_count = (df['loneliness_score'] > 0.7).sum()
            st.metric("High Risk Cases", high_risk_count)
        
        with col4:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Detailed history table
        st.markdown("### 📋 Detailed History")
        
        # Display table with formatting
        display_df = df.copy()
        display_df['loneliness_score'] = display_df['loneliness_score'].round(3)
        display_df['confidence'] = display_df['confidence'].round(3)
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        if st.button("📥 Download History"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="loneliness_analysis_history.csv",
                mime="text/csv"
            )
    
    def about_tab(self):
        """About and help tab"""
        st.markdown("## ℹ️ About the AI Elder Care System")
        
        st.markdown("""
        ### 🎯 Purpose
        This AI system is designed to detect early signs of loneliness in elderly individuals through analysis of 
        speech patterns and text content. It aims to support healthcare providers and family members in identifying 
        when intervention may be beneficial.
        
        ### 🧠 How It Works
        The system uses advanced machine learning techniques:
        
        - **Speech Analysis**: Analyzes prosodic features, voice quality, and emotional indicators
        - **Text Analysis**: Examines sentiment, linguistic patterns, and semantic content
        - **Ensemble Model**: Combines BERT for text and CNN/RNN for speech with attention mechanisms
        - **Explainable AI**: Provides clear explanations using SHAP, LIME, and attention visualization
        
        ### 📊 Understanding the Results
        
        **Loneliness Score**: A value between 0.0 and 1.0
        - 0.0 - 0.3: Low risk of loneliness
        - 0.3 - 0.6: Moderate risk
        - 0.6 - 0.8: High risk
        - 0.8 - 1.0: Very high risk
        
        **Confidence**: How certain the AI is about its prediction (higher is better)
        
        **Risk Level**: Categorical assessment (Low, Moderate, High, Very High)
        
        ### ⚠️ Important Disclaimers
        
        - This system is a **supportive tool**, not a replacement for professional medical assessment
        - Results should be interpreted by qualified healthcare professionals
        - The system is designed to identify potential concerns, not make diagnoses
        - Individual privacy and data security are paramount
        
        ### 🔒 Privacy & Security
        
        - No audio or text data is permanently stored
        - All analysis is performed locally when possible
        - Results are only retained in your current session
        - No personal information is transmitted or stored
        
        ### 🆘 Getting Help
        
        If you have questions about using this system or interpreting results:
        
        1. Consult the documentation
        2. Contact your healthcare provider
        3. Reach out to technical support
        
        ### 🔬 Research Background
        
        This system is based on cutting-edge research in:
        - Multimodal machine learning
        - Speech emotion recognition
        - Natural language processing
        - Explainable artificial intelligence
        - Elderly care and mental health
        """)

# Run the application
if __name__ == "__main__":
    app = LonelinessDetectionApp()
    app.run()
