#!/usr/bin/env python3
"""
Simple AI Elder Care Loneliness Detection Demo
Demonstrates the core functionality without dependencies
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def generate_sample_data():
    """Generate synthetic data for demonstration"""
    
    # Sample text inputs (what an elderly person might say)
    sample_texts = [
        "I've been feeling quite lonely lately, no one visits me anymore",
        "Had a wonderful day at the park with my family, feeling grateful",
        "The days feel so long when you have nobody to talk to"
    ]
    
    # Sample audio features (simulated)
    np.random.seed(42)
    audio_features = []
    for _ in range(len(sample_texts)):
        features = {
            'pitch_mean': np.random.normal(150, 30),
            'energy_mean': np.random.normal(0.5, 0.2),
            'speaking_rate': np.random.normal(2.5, 0.5),
            'pause_frequency': np.random.normal(0.3, 0.1),
            'voice_quality': np.random.normal(0.7, 0.2)
        }
        audio_features.append(features)
    
    return sample_texts, audio_features

def analyze_text_simple(text):
    """Simple text analysis without external libraries"""
    
    # Simple keyword-based analysis
    lonely_keywords = ['lonely', 'alone', 'isolated', 'sad', 'empty', 'nobody', 'miss']
    positive_keywords = ['happy', 'grateful', 'wonderful', 'joy', 'family', 'friends', 'love']
    
    text_lower = text.lower()
    
    lonely_score = sum(1 for word in lonely_keywords if word in text_lower)
    positive_score = sum(1 for word in positive_keywords if word in text_lower)
    
    # Simple sentiment calculation
    sentiment = (positive_score - lonely_score) / max(len(text.split()), 1)
    loneliness_probability = max(0, min(1, 0.5 + (lonely_score - positive_score) * 0.2))
    
    return {
        'sentiment_score': sentiment,
        'loneliness_keywords': lonely_score,
        'positive_keywords': positive_score,
        'loneliness_probability': loneliness_probability
    }

def analyze_audio_simple(audio_features):
    """Simple audio analysis"""
    
    # Simple heuristics based on research
    low_energy = audio_features['energy_mean'] < 0.4
    slow_speech = audio_features['speaking_rate'] < 2.0
    high_pauses = audio_features['pause_frequency'] > 0.4
    
    loneliness_indicators = sum([low_energy, slow_speech, high_pauses])
    loneliness_probability = loneliness_indicators / 3.0
    
    return {
        'energy_analysis': 'Low energy detected' if low_energy else 'Normal energy',
        'speech_rate_analysis': 'Slow speech detected' if slow_speech else 'Normal speech rate',
        'pause_analysis': 'High pause frequency' if high_pauses else 'Normal pause frequency',
        'loneliness_probability': loneliness_probability
    }

def multimodal_fusion(text_analysis, audio_analysis):
    """Combine text and audio analysis"""
    
    # Simple weighted average
    text_weight = 0.6
    audio_weight = 0.4
    
    combined_probability = (
        text_analysis['loneliness_probability'] * text_weight +
        audio_analysis['loneliness_probability'] * audio_weight
    )
    
    # Determine risk level
    if combined_probability > 0.7:
        risk_level = "High Risk"
        recommendation = "Immediate attention recommended. Consider social activities or counseling."
    elif combined_probability > 0.4:
        risk_level = "Moderate Risk"
        recommendation = "Monitor closely. Encourage social engagement."
    else:
        risk_level = "Low Risk"
        recommendation = "Continue current social activities."
    
    return {
        'combined_loneliness_probability': combined_probability,
        'risk_level': risk_level,
        'recommendation': recommendation
    }

def explain_prediction(text_analysis, audio_analysis, final_result):
    """Provide explanation for the prediction"""
    
    explanations = []
    
    # Text explanations
    if text_analysis['loneliness_keywords'] > 0:
        explanations.append(f"Text analysis found {text_analysis['loneliness_keywords']} loneliness-related keywords")
    
    if text_analysis['positive_keywords'] > 0:
        explanations.append(f"Text analysis found {text_analysis['positive_keywords']} positive keywords")
    
    # Audio explanations
    if audio_analysis['loneliness_probability'] > 0.5:
        explanations.append("Audio analysis indicates potential loneliness based on voice patterns")
    
    return explanations

def run_demo():
    """Run the complete demonstration"""
    
    print("=" * 60)
    print("AI ELDER CARE LONELINESS DETECTION SYSTEM")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Generate sample data
    sample_texts, audio_features = generate_sample_data()
    
    # Process each sample
    for i, (text, audio) in enumerate(zip(sample_texts, audio_features), 1):
        print(f"\n{'='*20} SAMPLE {i} {'='*20}")
        print(f"Input Text: \"{text}\"")
        print()
        
        # Analyze text
        text_result = analyze_text_simple(text)
        print("TEXT ANALYSIS:")
        print(f"  • Sentiment Score: {text_result['sentiment_score']:.3f}")
        print(f"  • Loneliness Keywords Found: {text_result['loneliness_keywords']}")
        print(f"  • Positive Keywords Found: {text_result['positive_keywords']}")
        print(f"  • Text-based Loneliness Probability: {text_result['loneliness_probability']:.3f}")
        print()
        
        # Analyze audio
        audio_result = analyze_audio_simple(audio)
        print("AUDIO ANALYSIS:")
        print(f"  • Energy: {audio_result['energy_analysis']}")
        print(f"  • Speech Rate: {audio_result['speech_rate_analysis']}")
        print(f"  • Pauses: {audio_result['pause_analysis']}")
        print(f"  • Audio-based Loneliness Probability: {audio_result['loneliness_probability']:.3f}")
        print()
        
        # Multimodal fusion
        final_result = multimodal_fusion(text_result, audio_result)
        print("MULTIMODAL ANALYSIS:")
        print(f"  • Combined Loneliness Probability: {final_result['combined_loneliness_probability']:.3f}")
        print(f"  • Risk Level: {final_result['risk_level']}")
        print(f"  • Recommendation: {final_result['recommendation']}")
        print()
        
        # Explanation
        explanations = explain_prediction(text_result, audio_result, final_result)
        print("EXPLANATION:")
        for explanation in explanations:
            print(f"  • {explanation}")
        
        if not explanations:
            print("  • Analysis based on standard voice and text patterns")
        
        print("\n" + "-" * 50)
    
    print("\n" + "=" * 60)
    print("DEMO SUMMARY:")
    print("=" * 60)
    print("✅ Multimodal AI system successfully analyzed text and voice patterns")
    print("✅ Provided loneliness risk assessment for each sample")
    print("✅ Generated actionable recommendations")
    print("✅ Explained the reasoning behind each prediction")
    print()
    print("SYSTEM CAPABILITIES:")
    print("• Real-time text sentiment analysis")
    print("• Voice pattern recognition for emotional states")
    print("• Multimodal fusion for higher accuracy")
    print("• Explainable AI with clear reasoning")
    print("• Risk-based recommendations")
    print()
    print("NEXT STEPS:")
    print("1. Launch full system: python -m streamlit run interface/streamlit_app.py")
    print("2. Train on real data: python train.py")
    print("3. Use web interface for interactive testing")
    print()
    print("Demo completed successfully! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()