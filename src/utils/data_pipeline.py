"""
Data Processing and Pipeline Management
Handles data loading, preprocessing, augmentation, and pipeline orchestration
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ..features import SpeechFeatureExtractor, TextFeatureExtractor, MultimodalFeatureFusion
from ..models import LonelinessDataset

class DataPreprocessor:
    """
    Comprehensive data preprocessing for multimodal loneliness detection
    """
    
    def __init__(self, 
                 speech_sample_rate: int = 16000,
                 text_max_length: int = 512,
                 normalize_features: bool = True):
        
        self.speech_sample_rate = speech_sample_rate
        self.text_max_length = text_max_length
        self.normalize_features = normalize_features
        
        # Initialize feature extractors
        self.speech_extractor = SpeechFeatureExtractor(sample_rate=speech_sample_rate)
        self.text_extractor = TextFeatureExtractor()
        self.fusion_processor = MultimodalFeatureFusion()
        
        # Scalers for feature normalization
        self.speech_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess_data(self, 
                                data_config: Dict,
                                cache_dir: Optional[str] = None) -> Dict[str, List]:
        """
        Load and preprocess multimodal data
        
        Args:
            data_config: Configuration dictionary with data paths and parameters
            cache_dir: Directory to cache preprocessed features
            
        Returns:
            Dictionary with preprocessed data
        """
        self.logger.info("Starting data loading and preprocessing...")
        
        # Create cache directory if specified
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "preprocessed_data.pkl")
            
            # Check if cached data exists
            if os.path.exists(cache_file):
                self.logger.info("Loading cached preprocessed data...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # Load raw data
        raw_data = self._load_raw_data(data_config)
        
        # Extract features
        processed_data = self._extract_all_features(raw_data)
        
        # Normalize features if requested
        if self.normalize_features:
            processed_data = self._normalize_features(processed_data)
        
        # Cache processed data
        if cache_dir:
            self.logger.info("Caching preprocessed data...")
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
        
        self.logger.info("Data preprocessing completed!")
        return processed_data
    
    def _load_raw_data(self, data_config: Dict) -> Dict[str, List]:
        """Load raw audio files, texts, and labels"""
        
        # Data can come from various sources
        if 'csv_file' in data_config:
            return self._load_from_csv(data_config['csv_file'])
        elif 'directory' in data_config:
            return self._load_from_directory(data_config['directory'])
        elif 'synthetic' in data_config:
            return self._generate_synthetic_data(data_config['synthetic'])
        else:
            raise ValueError("No valid data source specified in data_config")
    
    def _load_from_csv(self, csv_path: str) -> Dict[str, List]:
        """Load data from CSV file"""
        df = pd.read_csv(csv_path)
        
        required_columns = ['audio_path', 'text', 'loneliness_score']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        return {
            'audio_paths': df['audio_path'].tolist(),
            'texts': df['text'].tolist(),
            'labels': df['loneliness_score'].tolist(),
            'metadata': df.drop(required_columns, axis=1).to_dict('records') if len(df.columns) > 3 else []
        }
    
    def _load_from_directory(self, directory_path: str) -> Dict[str, List]:
        """Load data from organized directory structure"""
        directory = Path(directory_path)
        
        audio_paths = []
        texts = []
        labels = []
        metadata = []
        
        # Expected structure: directory/class/audio_files with corresponding text files
        for class_dir in directory.iterdir():
            if class_dir.is_dir():
                label = float(class_dir.name)  # Directory name as loneliness score
                
                for audio_file in class_dir.glob("*.wav"):
                    text_file = audio_file.with_suffix('.txt')
                    
                    if text_file.exists():
                        audio_paths.append(str(audio_file))
                        
                        with open(text_file, 'r', encoding='utf-8') as f:
                            texts.append(f.read().strip())
                        
                        labels.append(label)
                        metadata.append({'source_dir': class_dir.name})
        
        return {
            'audio_paths': audio_paths,
            'texts': texts,
            'labels': labels,
            'metadata': metadata
        }
    
    def _generate_synthetic_data(self, config: Dict) -> Dict[str, List]:
        """Generate synthetic data for testing purposes"""
        n_samples = config.get('n_samples', 100)
        
        self.logger.info(f"Generating {n_samples} synthetic samples...")
        
        # Generate synthetic text data
        lonely_phrases = [
            "I feel so alone these days. Nobody calls me anymore.",
            "The house is so quiet. I miss having people around.",
            "I haven't spoken to anyone in days. It's very isolating.",
            "Sometimes I feel forgotten by everyone I used to know.",
            "The silence is overwhelming. I wish I had someone to talk to."
        ]
        
        not_lonely_phrases = [
            "I had a wonderful visit from my grandchildren today.",
            "My friends and I went out for coffee this morning.",
            "I'm involved in several community activities and feel connected.",
            "My family calls regularly and we have great conversations.",
            "I feel blessed to have such a supportive social network."
        ]
        
        audio_paths = []
        texts = []
        labels = []
        
        for i in range(n_samples):
            # Randomly assign loneliness level
            is_lonely = np.random.random() > 0.5
            loneliness_score = np.random.uniform(0.6, 1.0) if is_lonely else np.random.uniform(0.0, 0.4)
            
            # Select appropriate text
            if is_lonely:
                text = np.random.choice(lonely_phrases)
            else:
                text = np.random.choice(not_lonely_phrases)
            
            # Add some noise to text
            text += f" Today is a {'difficult' if is_lonely else 'good'} day for me."
            
            # Generate synthetic audio path (we'll create dummy features)
            audio_path = f"synthetic_audio_{i}.wav"
            
            audio_paths.append(audio_path)
            texts.append(text)
            labels.append(loneliness_score)
        
        return {
            'audio_paths': audio_paths,
            'texts': texts,
            'labels': labels,
            'metadata': [{'synthetic': True} for _ in range(n_samples)]
        }
    
    def _extract_all_features(self, raw_data: Dict) -> Dict[str, List]:
        """Extract features from all modalities"""
        audio_paths = raw_data['audio_paths']
        texts = raw_data['texts']
        labels = raw_data['labels']
        
        speech_features = []
        text_features = []
        valid_indices = []
        
        self.logger.info("Extracting features from audio and text...")
        
        for i, (audio_path, text) in enumerate(tqdm(zip(audio_paths, texts), total=len(audio_paths))):
            try:
                # Extract speech features
                if audio_path.startswith('synthetic_'):
                    # Generate synthetic speech features
                    speech_feat = self._generate_synthetic_speech_features(labels[i])
                else:
                    if os.path.exists(audio_path):
                        speech_feat = self.speech_extractor.extract_features(audio_path)['combined']
                    else:
                        self.logger.warning(f"Audio file not found: {audio_path}, generating synthetic features")
                        speech_feat = self._generate_synthetic_speech_features(labels[i])
                
                # Extract text features
                text_feat = self.text_extractor.extract_features(text)['combined']
                
                speech_features.append(speech_feat)
                text_features.append(text_feat)
                valid_indices.append(i)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Filter labels and metadata for valid samples
        valid_labels = [labels[i] for i in valid_indices]
        valid_metadata = [raw_data.get('metadata', [{}] * len(labels))[i] for i in valid_indices]
        valid_texts = [texts[i] for i in valid_indices]
        
        return {
            'speech_features': speech_features,
            'text_features': text_features,
            'texts': valid_texts,
            'labels': valid_labels,
            'metadata': valid_metadata
        }
    
    def _generate_synthetic_speech_features(self, loneliness_score: float) -> np.ndarray:
        """Generate synthetic speech features based on loneliness score"""
        # Create realistic synthetic features that correlate with loneliness
        
        base_features = np.random.randn(128) * 0.1
        
        # Adjust features based on loneliness score
        if loneliness_score > 0.5:  # Lonely
            # Lower pitch variation, slower speech, more pauses
            base_features[1] *= 0.5  # Reduced pitch variation
            base_features[7] *= 0.7  # Slower speech rate
            base_features[5] += 0.3  # More pauses
            base_features[42] -= 0.2  # Lower energy
        else:  # Not lonely
            # More animated speech patterns
            base_features[1] *= 1.3  # More pitch variation
            base_features[7] *= 1.2  # Faster speech rate
            base_features[42] += 0.2  # Higher energy
        
        # Add some noise
        base_features += np.random.randn(128) * 0.05
        
        return base_features
    
    def _normalize_features(self, data: Dict) -> Dict:
        """Normalize features using standard scaling"""
        self.logger.info("Normalizing features...")
        
        speech_features = np.array(data['speech_features'])
        text_features = np.array(data['text_features'])
        
        # Fit and transform scalers
        normalized_speech = self.speech_scaler.fit_transform(speech_features)
        normalized_text = self.text_scaler.fit_transform(text_features)
        
        # Update data
        data['speech_features'] = normalized_speech.tolist()
        data['text_features'] = normalized_text.tolist()
        
        return data
    
    def create_data_splits(self, 
                          data: Dict,
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          stratify: bool = True,
                          random_state: int = 42) -> Dict[str, Dict]:
        """Create train/validation/test splits"""
        
        self.logger.info("Creating data splits...")
        
        speech_features = data['speech_features']
        text_features = data['text_features']
        texts = data['texts']
        labels = data['labels']
        
        # Convert continuous labels to categorical for stratification
        if stratify:
            # Create bins for stratification
            label_bins = pd.cut(labels, bins=3, labels=['low', 'medium', 'high'])
        else:
            label_bins = None
        
        # First split: train+val vs test
        indices = list(range(len(labels)))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, 
            stratify=label_bins if stratify else None,
            random_state=random_state
        )
        
        # Second split: train vs val
        if stratify:
            train_val_bins = [label_bins[i] for i in train_val_idx]
        else:
            train_val_bins = None
        
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size/(1-test_size),
            stratify=train_val_bins if stratify else None,
            random_state=random_state
        )
        
        # Create split dictionaries
        def create_split_data(indices):
            return {
                'speech_features': [speech_features[i] for i in indices],
                'text_features': [text_features[i] for i in indices],
                'texts': [texts[i] for i in indices],
                'labels': [labels[i] for i in indices]
            }
        
        splits = {
            'train': create_split_data(train_idx),
            'val': create_split_data(val_idx),
            'test': create_split_data(test_idx)
        }
        
        # Log split statistics
        for split_name, split_data in splits.items():
            n_samples = len(split_data['labels'])
            mean_label = np.mean(split_data['labels'])
            self.logger.info(f"{split_name}: {n_samples} samples, mean loneliness: {mean_label:.3f}")
        
        return splits

class DataAugmentation:
    """
    Data augmentation techniques for multimodal data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def augment_speech_features(self, 
                               speech_features: np.ndarray,
                               augmentation_factor: float = 0.1) -> List[np.ndarray]:
        """Augment speech features with noise and variations"""
        
        augmented_features = []
        
        # Original features
        augmented_features.append(speech_features)
        
        # Add Gaussian noise
        noise_variant = speech_features + np.random.normal(0, augmentation_factor, speech_features.shape)
        augmented_features.append(noise_variant)
        
        # Scale features slightly
        scale_factor = np.random.uniform(0.9, 1.1)
        scaled_variant = speech_features * scale_factor
        augmented_features.append(scaled_variant)
        
        # Time shift simulation (permute some features)
        shifted_variant = speech_features.copy()
        n_features_to_shift = int(len(speech_features) * 0.1)
        shift_indices = np.random.choice(len(speech_features), n_features_to_shift, replace=False)
        shifted_variant[shift_indices] += np.random.normal(0, augmentation_factor, n_features_to_shift)
        augmented_features.append(shifted_variant)
        
        return augmented_features
    
    def augment_text(self, text: str) -> List[str]:
        """Augment text with paraphrasing and synonym replacement"""
        augmented_texts = [text]  # Original text
        
        # Simple text augmentations
        # Add filler words
        filler_words = ['um', 'uh', 'you know', 'like', 'well']
        words = text.split()
        if len(words) > 2:
            insert_pos = np.random.randint(1, len(words))
            filler = np.random.choice(filler_words)
            augmented_words = words[:insert_pos] + [filler] + words[insert_pos:]
            augmented_texts.append(' '.join(augmented_words))
        
        # Add emotional emphasis
        if any(word in text.lower() for word in ['sad', 'lonely', 'alone', 'isolated']):
            emphasized_text = text + " It's really hard for me."
            augmented_texts.append(emphasized_text)
        
        # Add hesitation
        hesitant_text = text.replace('.', '... ').replace('!', '... ')
        if hesitant_text != text:
            augmented_texts.append(hesitant_text)
        
        return augmented_texts

class DataPipeline:
    """
    Complete data pipeline orchestration
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = DataPreprocessor(
            speech_sample_rate=config.get('speech_sample_rate', 16000),
            text_max_length=config.get('text_max_length', 512),
            normalize_features=config.get('normalize_features', True)
        )
        self.augmentation = DataAugmentation()
        
        self.logger = logging.getLogger(__name__)
    
    def run_pipeline(self) -> Dict[str, DataLoader]:
        """Run the complete data pipeline"""
        
        self.logger.info("Starting complete data pipeline...")
        
        # Load and preprocess data
        processed_data = self.preprocessor.load_and_preprocess_data(
            self.config['data_config'],
            cache_dir=self.config.get('cache_dir')
        )
        
        # Create data splits
        splits = self.preprocessor.create_data_splits(
            processed_data,
            test_size=self.config.get('test_size', 0.2),
            val_size=self.config.get('val_size', 0.1),
            stratify=self.config.get('stratify', True),
            random_state=self.config.get('random_state', 42)
        )
        
        # Apply data augmentation if specified
        if self.config.get('apply_augmentation', False):
            splits['train'] = self._apply_augmentation(splits['train'])
        
        # Create DataLoaders
        dataloaders = {}
        for split_name, split_data in splits.items():
            dataset = LonelinessDataset(
                speech_features=split_data['speech_features'],
                texts=split_data['texts'],
                labels=split_data['labels'],
                tokenizer=None,  # Will be set by trainer
                max_length=self.config.get('text_max_length', 512)
            )
            
            batch_size = self.config.get('batch_size', 32)
            shuffle = split_name == 'train'
            
            dataloaders[split_name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=self.config.get('num_workers', 0),
                pin_memory=torch.cuda.is_available()
            )
        
        self.logger.info("Data pipeline completed successfully!")
        return dataloaders
    
    def _apply_augmentation(self, train_data: Dict) -> Dict:
        """Apply data augmentation to training data"""
        
        self.logger.info("Applying data augmentation...")
        
        augmented_speech = []
        augmented_texts = []
        augmented_labels = []
        
        for speech_feat, text, label in zip(train_data['speech_features'], 
                                           train_data['texts'], 
                                           train_data['labels']):
            
            # Augment speech features
            speech_variants = self.augmentation.augment_speech_features(np.array(speech_feat))
            
            # Augment text
            text_variants = self.augmentation.augment_text(text)
            
            # Create all combinations
            for speech_var in speech_variants:
                for text_var in text_variants:
                    augmented_speech.append(speech_var.tolist())
                    augmented_texts.append(text_var)
                    augmented_labels.append(label)
        
        augmented_data = {
            'speech_features': augmented_speech,
            'text_features': train_data['text_features'] * len(augmented_speech) // len(train_data['speech_features']),
            'texts': augmented_texts,
            'labels': augmented_labels
        }
        
        self.logger.info(f"Augmentation complete: {len(train_data['labels'])} -> {len(augmented_labels)} samples")
        
        return augmented_data
