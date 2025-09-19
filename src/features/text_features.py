"""
Text Feature Extraction Module
Extracts linguistic, semantic, and emotional features from text
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import re
from collections import Counter

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

class TextFeatureExtractor:
    """
    Advanced text feature extraction for loneliness detection
    Focuses on sentiment, linguistic patterns, and semantic features
    """
    
    def __init__(self):
        # Initialize NLP tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Loneliness-related keywords
        self.loneliness_keywords = {
            'isolation': ['alone', 'lonely', 'isolated', 'solitary', 'abandoned', 'forgotten'],
            'sadness': ['sad', 'depressed', 'down', 'blue', 'melancholy', 'gloomy'],
            'social_withdrawal': ['avoid', 'withdraw', 'hide', 'retreat', 'distance'],
            'negative_emotion': ['empty', 'hollow', 'void', 'meaningless', 'hopeless'],
            'social_connection': ['friends', 'family', 'people', 'visit', 'call', 'together'],
            'positive_emotion': ['happy', 'joy', 'love', 'content', 'grateful', 'blessed']
        }
        
        self.feature_dim = 85  # Total feature dimension
    
    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive text features
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing various feature types
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Extract different feature types
        features = {
            'sentiment': self._extract_sentiment_features(text),
            'linguistic': self._extract_linguistic_features(text),
            'semantic': self._extract_semantic_features(text),
            'emotional': self._extract_emotional_features(text),
            'syntactic': self._extract_syntactic_features(text),
            'lexical': self._extract_lexical_features(text)
        }
        
        # Combine all features
        combined_features = np.concatenate([
            features['sentiment'],
            features['linguistic'],
            features['semantic'],
            features['emotional'],
            features['syntactic'],
            features['lexical']
        ])
        
        features['combined'] = combined_features
        return features
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove special characters but keep punctuation for sentiment analysis
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text
    
    def _extract_sentiment_features(self, text: str) -> np.ndarray:
        """Extract sentiment-related features"""
        features = []
        
        # VADER sentiment analysis
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        features.extend([
            vader_scores['pos'],      # Positive sentiment
            vader_scores['neu'],      # Neutral sentiment  
            vader_scores['neg'],      # Negative sentiment
            vader_scores['compound']  # Compound sentiment
        ])
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        features.extend([
            blob.sentiment.polarity,    # Polarity (-1 to 1)
            blob.sentiment.subjectivity # Subjectivity (0 to 1)
        ])
        
        # Sentence-level sentiment variation
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            sent_polarities = [TextBlob(sent).sentiment.polarity for sent in sentences]
            features.extend([
                np.mean(sent_polarities),  # Mean sentiment
                np.std(sent_polarities),   # Sentiment variation
                np.min(sent_polarities),   # Most negative
                np.max(sent_polarities)    # Most positive
            ])
        else:
            features.extend([blob.sentiment.polarity, 0, blob.sentiment.polarity, blob.sentiment.polarity])
        
        return np.array(features)
    
    def _extract_linguistic_features(self, text: str) -> np.ndarray:
        """Extract linguistic and structural features"""
        features = []
        
        # Basic text statistics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        features.extend([
            len(words),                    # Word count
            len(sentences),                # Sentence count
            len(words) / max(len(sentences), 1),  # Average words per sentence
            len([w for w in words if len(w) > 6]),  # Long words count
            len(set(words)) / max(len(words), 1)    # Lexical diversity
        ])
        
        # Character-level features
        features.extend([
            len(text),                     # Character count
            len(text) / max(len(words), 1), # Average word length
            text.count('!'),               # Exclamation marks
            text.count('?'),               # Question marks
            text.count(','),               # Commas
            text.count('.')                # Periods
        ])
        
        # Part-of-speech features
        pos_tags = pos_tag(words)
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        # Key POS categories
        features.extend([
            pos_counts.get('NN', 0) + pos_counts.get('NNS', 0),   # Nouns
            pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0),  # Verbs
            pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0),   # Adjectives
            pos_counts.get('RB', 0) + pos_counts.get('RBR', 0),   # Adverbs
            pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)  # Pronouns
        ])
        
        return np.array(features)
    
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """Extract semantic and meaning-related features"""
        features = []
        
        # Loneliness-related keyword analysis
        words = word_tokenize(text.lower())
        
        for category, keywords in self.loneliness_keywords.items():
            # Count occurrences of keywords in each category
            count = sum(1 for word in words if any(kw in word for kw in keywords))
            features.append(count / max(len(words), 1))  # Normalized frequency
        
        # First-person vs. third-person reference
        first_person = ['i', 'me', 'my', 'myself', 'mine']
        third_person = ['he', 'she', 'they', 'them', 'his', 'her', 'their']
        
        first_person_count = sum(1 for word in words if word in first_person)
        third_person_count = sum(1 for word in words if word in third_person)
        
        features.extend([
            first_person_count / max(len(words), 1),    # First-person density
            third_person_count / max(len(words), 1),    # Third-person density
        ])
        
        # Time references
        past_indicators = ['was', 'were', 'had', 'did', 'yesterday', 'before', 'ago']
        present_indicators = ['am', 'is', 'are', 'have', 'do', 'today', 'now', 'currently']
        future_indicators = ['will', 'shall', 'going', 'tomorrow', 'later', 'soon']
        
        features.extend([
            sum(1 for word in words if word in past_indicators) / max(len(words), 1),
            sum(1 for word in words if word in present_indicators) / max(len(words), 1),
            sum(1 for word in words if word in future_indicators) / max(len(words), 1)
        ])
        
        return np.array(features)
    
    def _extract_emotional_features(self, text: str) -> np.ndarray:
        """Extract emotional indicators"""
        features = []
        
        # Emotional word categories
        emotional_words = {
            'anxiety': ['worried', 'anxious', 'nervous', 'scared', 'afraid', 'fearful'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
            'joy': ['happy', 'joyful', 'cheerful', 'delighted', 'pleased', 'glad'],
            'trust': ['trust', 'faith', 'confidence', 'believe', 'reliable', 'secure'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'unexpected'],
            'disgust': ['disgusted', 'revolted', 'sickened', 'appalled'],
            'anticipation': ['excited', 'eager', 'hopeful', 'optimistic', 'looking forward']
        }
        
        words = word_tokenize(text.lower())
        
        for emotion, emotion_words_list in emotional_words.items():
            count = sum(1 for word in words if any(ew in word for ew in emotion_words_list))
            features.append(count / max(len(words), 1))
        
        # Intensity modifiers
        intensifiers = ['very', 'extremely', 'really', 'quite', 'so', 'too', 'completely']
        diminishers = ['slightly', 'somewhat', 'rather', 'little', 'bit', 'barely']
        
        features.extend([
            sum(1 for word in words if word in intensifiers) / max(len(words), 1),
            sum(1 for word in words if word in diminishers) / max(len(words), 1)
        ])
        
        return np.array(features)
    
    def _extract_syntactic_features(self, text: str) -> np.ndarray:
        """Extract syntactic complexity features"""
        features = []
        
        if self.nlp is not None:
            doc = self.nlp(text)
            
            # Dependency parsing features
            features.extend([
                len([token for token in doc if token.dep_ == 'nsubj']) / max(len(doc), 1),  # Subject frequency
                len([token for token in doc if token.dep_ == 'dobj']) / max(len(doc), 1),   # Direct object frequency
                len([token for token in doc if token.dep_ == 'prep']) / max(len(doc), 1),   # Preposition frequency
            ])
            
            # Named entity features
            entities = doc.ents
            features.extend([
                len(entities) / max(len(doc), 1),                                           # Entity density
                len([ent for ent in entities if ent.label_ == 'PERSON']) / max(len(doc), 1) # Person mentions
            ])
        else:
            # Fallback features when spaCy is not available
            features.extend([0, 0, 0, 0, 0])
        
        # Simple syntactic complexity measures
        sentences = sent_tokenize(text)
        if len(sentences) > 0:
            avg_sent_length = np.mean([len(word_tokenize(sent)) for sent in sentences])
            features.append(avg_sent_length)
        else:
            features.append(0)
        
        return np.array(features)
    
    def _extract_lexical_features(self, text: str) -> np.ndarray:
        """Extract lexical richness and complexity features"""
        features = []
        
        words = word_tokenize(text.lower())
        content_words = [w for w in words if w not in self.stop_words and w.isalpha()]
        
        if len(words) > 0:
            # Lexical density
            features.append(len(content_words) / len(words))
            
            # Function words ratio
            function_words = [w for w in words if w in self.stop_words]
            features.append(len(function_words) / len(words))
            
            # Average word frequency (using word length as proxy)
            avg_word_length = np.mean([len(w) for w in content_words]) if content_words else 0
            features.append(avg_word_length)
            
        else:
            features.extend([0, 0, 0])
        
        # Readability approximation (Flesch-Kincaid inspired)
        sentences = sent_tokenize(text)
        if len(sentences) > 0 and len(words) > 0:
            avg_sent_length = len(words) / len(sentences)
            # Simple syllable count approximation
            syllable_count = sum([max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in words])
            avg_syllables = syllable_count / len(words)
            readability_score = 206.835 - (1.015 * avg_sent_length) - (84.6 * avg_syllables)
            features.append(max(0, min(100, readability_score)))  # Clip to 0-100 range
        else:
            features.append(50)  # Default moderate readability
        
        return np.array(features)
    
    def extract_features_batch(self, texts: List[str]) -> np.ndarray:
        """Extract features from multiple texts"""
        features_list = []
        for text in texts:
            try:
                features = self.extract_features(text)
                features_list.append(features['combined'])
            except Exception as e:
                print(f"Error processing text: {e}")
                # Add zero features for failed extractions
                features_list.append(np.zeros(self.feature_dim))
        
        return np.array(features_list)
