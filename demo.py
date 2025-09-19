"""
Demo Script for Loneliness Detection System
Quick demonstration of the complete system
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models import LonelinessDetectionModel
from src.features import SpeechFeatureExtractor, TextFeatureExtractor
from src.explainability import ModelExplainer, ReportGenerator
from transformers import AutoTokenizer

def run_demo():
    """Run a comprehensive demo of the system"""
    
    print("🤗 AI Elder Care - Loneliness Detection System Demo")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing AI components...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Using device: {device}")
    
    # Load model (demo version with random weights)
    model = LonelinessDetectionModel()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Feature extractors
    speech_extractor = SpeechFeatureExtractor()
    text_extractor = TextFeatureExtractor()
    
    # Explainer
    explainer = ModelExplainer(model, tokenizer, device)
    report_generator = ReportGenerator()
    
    print("✅ Components initialized successfully!")
    print()
    
    # Demo scenarios
    scenarios = [
        {
            "name": "High Loneliness Case",
            "text": "I feel so alone these days. Nobody calls me anymore, and I spend most of my time by myself. The house is so quiet and empty. I miss having people around to talk to.",
            "expected_score": "High"
        },
        {
            "name": "Low Loneliness Case", 
            "text": "I had a wonderful visit from my grandchildren today. We played games and had such a lovely time together. My family calls regularly and I feel very blessed to have such support.",
            "expected_score": "Low"
        },
        {
            "name": "Mixed Case",
            "text": "Some days are better than others. I try to stay positive, but sometimes the loneliness creeps in when it gets dark. I wish I had more social activities to look forward to.",
            "expected_score": "Moderate"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"🔍 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        print(f"📝 Text: {scenario['text']}")
        print()
        
        # Generate synthetic speech features based on text
        speech_features = generate_demo_speech_features(scenario['text'])
        
        # Extract text features
        text_features = text_extractor.extract_features(scenario['text'])
        print(f"📊 Extracted {len(text_features['combined'])} text features")
        print(f"🎵 Generated {len(speech_features)} speech features")
        print()
        
        # Get AI prediction with explanations
        print("🧠 AI Analysis in progress...")
        explanation = explainer.explain_prediction(
            speech_features, 
            scenario['text'], 
            explanation_types=['attention', 'shap']
        )
        
        # Display results
        prediction = explanation['prediction']
        loneliness_score = prediction['loneliness_score']
        confidence = prediction['confidence']
        risk_level = prediction['risk_level']
        
        print(f"📈 Loneliness Score: {loneliness_score:.3f}")
        print(f"🎯 Risk Level: {risk_level}")
        print(f"✨ Confidence: {confidence:.1%}")
        print()
        
        # Show explanation
        if 'natural_language' in explanation:
            print(f"💬 AI Explanation:")
            print(f"   {explanation['natural_language']}")
            print()
        
        # Show attention analysis
        if 'attention' in explanation and 'text_attention' in explanation['attention']:
            top_tokens = explanation['attention']['text_attention'].get('top_tokens', [])[:3]
            if top_tokens:
                print(f"🔍 Key words that influenced the prediction:")
                for token, weight in top_tokens:
                    print(f"   • {token}: {weight:.3f}")
                print()
        
        # Show feature importance
        if 'feature_importance' in explanation:
            contrib = explanation['feature_importance'].get('modality_contributions', {})
            if contrib:
                print(f"📊 Analysis contributions:")
                print(f"   • Speech: {contrib.get('speech', 0):.1%}")
                print(f"   • Text: {contrib.get('text', 0):.1%}")
                print()
        
        results.append({
            'scenario': scenario['name'],
            'text': scenario['text'],
            'loneliness_score': loneliness_score,
            'risk_level': risk_level,
            'confidence': confidence,
            'explanation': explanation
        })
        
        print("=" * 60)
        print()
    
    # Generate comprehensive report
    print("📋 Generating comprehensive analysis report...")
    
    # Create a sample explanation for report
    sample_explanation = results[0]['explanation']
    report = report_generator.generate_explanation_report(sample_explanation)
    
    # Save report
    os.makedirs('demo_results', exist_ok=True)
    report_path = 'demo_results/sample_analysis_report.md'
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📁 Report saved to: {report_path}")
    print()
    
    # Summary
    print("📊 DEMO SUMMARY")
    print("=" * 30)
    
    for result in results:
        print(f"• {result['scenario']}: {result['loneliness_score']:.2f} ({result['risk_level']})")
    
    print()
    print("✅ Demo completed successfully!")
    print()
    print("🚀 Next Steps:")
    print("   1. Train the model with real data using: python train.py --data-config synthetic")
    print("   2. Run the web interface: streamlit run interface/streamlit_app.py")
    print("   3. Explore the Gradio interface: python interface/gradio_app.py")
    print()
    print("📖 For more information, see README.md")

def generate_demo_speech_features(text: str) -> np.ndarray:
    """Generate realistic speech features for demo"""
    
    # Base random features
    features = np.random.randn(128) * 0.1
    
    # Analyze text sentiment to adjust speech features
    lonely_words = ['alone', 'lonely', 'isolated', 'sad', 'empty', 'quiet', 'miss']
    social_words = ['visit', 'family', 'friends', 'together', 'blessed', 'support', 'wonderful']
    negative_words = ['nobody', 'never', 'nothing', 'no one', 'empty', 'dark']
    positive_words = ['love', 'happy', 'joy', 'great', 'amazing', 'fantastic']
    
    text_lower = text.lower()
    
    # Count word categories
    lonely_count = sum(1 for word in lonely_words if word in text_lower)
    social_count = sum(1 for word in social_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    # Adjust prosodic features based on content
    if lonely_count > social_count or negative_count > positive_count:
        # Lonely/sad speech characteristics
        features[0] -= 0.3  # Lower mean pitch
        features[1] *= 0.6  # Reduced pitch variation
        features[7] *= 0.7  # Slower speech rate
        features[5] += 0.4  # More pauses
        features[23] -= 0.2  # Lower energy
        features[42] -= 0.3  # Reduced vocal energy
    else:
        # Happy/social speech characteristics  
        features[0] += 0.2  # Higher mean pitch
        features[1] *= 1.4  # More pitch variation
        features[7] *= 1.3  # Faster speech rate
        features[5] *= 0.6  # Fewer pauses
        features[23] += 0.3  # Higher energy
        features[42] += 0.2  # More vocal energy
    
    # Add some realistic noise
    features += np.random.randn(128) * 0.05
    
    return features

if __name__ == "__main__":
    run_demo()
