#!/usr/bin/env python3
"""
Interactive AI Elder Care Loneliness Detection
Run this for immediate testing with your own input!
"""

import sys

def analyze_text_simple(text):
    """Simple text analysis"""
    lonely_keywords = ['lonely', 'alone', 'isolated', 'sad', 'empty', 'nobody', 'miss', 'forgotten', 'silence', 'quiet']
    positive_keywords = ['happy', 'grateful', 'wonderful', 'joy', 'family', 'friends', 'love', 'community', 'together', 'visited']
    
    text_lower = text.lower()
    
    lonely_score = sum(1 for word in lonely_keywords if word in text_lower)
    positive_score = sum(1 for word in positive_keywords if word in text_lower)
    
    sentiment = (positive_score - lonely_score) / max(len(text.split()), 1)
    loneliness_probability = max(0, min(1, 0.5 + (lonely_score - positive_score) * 0.2))
    
    return {
        'sentiment_score': sentiment,
        'loneliness_keywords': lonely_score,
        'positive_keywords': positive_score,
        'loneliness_probability': loneliness_probability
    }

def get_risk_level(probability):
    """Determine risk level and recommendation"""
    if probability > 0.7:
        return "🔴 High Risk", "Immediate attention recommended. Consider social activities or counseling."
    elif probability > 0.4:
        return "🟡 Moderate Risk", "Monitor closely. Encourage social engagement."
    else:
        return "🟢 Low Risk", "Continue current social activities."

def interactive_demo():
    """Interactive demo where user can input their own text"""
    
    print("=" * 60)
    print("🤖 AI ELDER CARE LONELINESS DETECTION - INTERACTIVE MODE")
    print("=" * 60)
    print("Enter text that an elderly person might say, and get instant analysis!")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        print("💬 Enter text to analyze:")
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for testing the AI Elder Care system!")
            break
            
        if not user_input:
            print("❌ Please enter some text to analyze.\n")
            continue
            
        # Analyze the input
        print(f"\n🔍 Analyzing: \"{user_input}\"")
        print("-" * 50)
        
        result = analyze_text_simple(user_input)
        risk_level, recommendation = get_risk_level(result['loneliness_probability'])
        
        # Display results
        print("📊 ANALYSIS RESULTS:")
        print(f"   • Loneliness Probability: {result['loneliness_probability']:.3f}")
        print(f"   • Risk Level: {risk_level}")
        print(f"   • Loneliness Keywords Found: {result['loneliness_keywords']}")
        print(f"   • Positive Keywords Found: {result['positive_keywords']}")
        print(f"   • Sentiment Score: {result['sentiment_score']:.3f}")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   {recommendation}")
        
        print(f"\n🧠 EXPLANATION:")
        if result['loneliness_keywords'] > 0:
            print(f"   • Found {result['loneliness_keywords']} loneliness-related words")
        if result['positive_keywords'] > 0:
            print(f"   • Found {result['positive_keywords']} positive/social words")
        if result['loneliness_keywords'] == 0 and result['positive_keywords'] == 0:
            print(f"   • Analysis based on general language patterns")
            
        print("\n" + "=" * 60)
        print("Try another text or type 'quit' to exit")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    interactive_demo()