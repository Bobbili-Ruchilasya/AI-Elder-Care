"""
Evaluation Metrics and Assessment Framework
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import logging

class ModelEvaluator:
    """
    Comprehensive evaluation framework for loneliness detection model
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
    
    def evaluate_predictions(self, 
                           y_true: List[float], 
                           y_pred: List[float],
                           y_prob: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of model predictions
        
        Args:
            y_true: True loneliness scores
            y_pred: Predicted loneliness scores
            y_prob: Prediction probabilities (if different from y_pred)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_prob is None:
            y_prob = y_pred
        else:
            y_prob = np.array(y_prob)
        
        metrics = {}
        
        # Regression metrics (continuous scores)
        metrics.update(self._compute_regression_metrics(y_true, y_pred))
        
        # Classification metrics (binary)
        y_true_binary = (y_true > self.threshold).astype(int)
        y_pred_binary = (y_pred > self.threshold).astype(int)
        metrics.update(self._compute_classification_metrics(y_true_binary, y_pred_binary, y_prob))
        
        # Correlation metrics
        metrics.update(self._compute_correlation_metrics(y_true, y_pred))
        
        # Custom loneliness-specific metrics
        metrics.update(self._compute_loneliness_metrics(y_true, y_pred))
        
        return metrics
    
    def _compute_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression evaluation metrics"""
        metrics = {}
        
        # Mean Squared Error
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # R-squared Score
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return metrics
    
    def _compute_classification_metrics(self, 
                                      y_true_binary: np.ndarray, 
                                      y_pred_binary: np.ndarray,
                                      y_prob: np.ndarray) -> Dict[str, float]:
        """Compute classification evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Area Under ROC Curve
        try:
            metrics['auc_roc'] = roc_auc_score(y_true_binary, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.5  # Random performance if only one class
        
        # Specificity and Sensitivity
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balanced Accuracy
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        return metrics
    
    def _compute_correlation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute correlation metrics"""
        metrics = {}
        
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        metrics['pearson_correlation'] = pearson_corr
        metrics['pearson_p_value'] = pearson_p
        
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        metrics['spearman_correlation'] = spearman_corr
        metrics['spearman_p_value'] = spearman_p
        
        return metrics
    
    def _compute_loneliness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute loneliness-specific evaluation metrics"""
        metrics = {}
        
        # Risk level accuracy
        true_risk_levels = self._get_risk_levels(y_true)
        pred_risk_levels = self._get_risk_levels(y_pred)
        metrics['risk_level_accuracy'] = accuracy_score(true_risk_levels, pred_risk_levels)
        
        # High-risk detection metrics
        high_risk_true = (y_true >= 0.7).astype(int)
        high_risk_pred = (y_pred >= 0.7).astype(int)
        
        if np.sum(high_risk_true) > 0:
            metrics['high_risk_precision'] = precision_recall_fscore_support(
                high_risk_true, high_risk_pred, average='binary', zero_division=0
            )[0]
            metrics['high_risk_recall'] = precision_recall_fscore_support(
                high_risk_true, high_risk_pred, average='binary', zero_division=0
            )[1]
        else:
            metrics['high_risk_precision'] = 0
            metrics['high_risk_recall'] = 0
        
        # Early detection capability (for scores > 0.3)
        early_detection_true = (y_true >= 0.3).astype(int)
        early_detection_pred = (y_pred >= 0.3).astype(int)
        metrics['early_detection_accuracy'] = accuracy_score(early_detection_true, early_detection_pred)
        
        return metrics
    
    def _get_risk_levels(self, scores: np.ndarray) -> List[str]:
        """Convert continuous scores to risk levels"""
        risk_levels = []
        for score in scores:
            if score < 0.3:
                risk_levels.append('Low')
            elif score < 0.6:
                risk_levels.append('Moderate')
            elif score < 0.8:
                risk_levels.append('High')
            else:
                risk_levels.append('Very High')
        return risk_levels
    
    def create_evaluation_report(self, 
                               y_true: List[float],
                               y_pred: List[float],
                               model_name: str = "Loneliness Detection Model") -> str:
        """Generate comprehensive evaluation report"""
        
        metrics = self.evaluate_predictions(y_true, y_pred)
        
        report = []
        report.append(f"# {model_name} - Evaluation Report\n")
        report.append("=" * 50 + "\n")
        
        # Model Performance Summary
        report.append("## Performance Summary\n")
        report.append(f"- **Overall Accuracy**: {metrics['accuracy']:.3f}")
        report.append(f"- **F1-Score**: {metrics['f1_score']:.3f}")
        report.append(f"- **AUC-ROC**: {metrics['auc_roc']:.3f}")
        report.append(f"- **RMSE**: {metrics['rmse']:.3f}")
        report.append(f"- **Pearson Correlation**: {metrics['pearson_correlation']:.3f}\n")
        
        # Classification Metrics
        report.append("## Classification Metrics\n")
        report.append(f"- **Precision**: {metrics['precision']:.3f}")
        report.append(f"- **Recall (Sensitivity)**: {metrics['recall']:.3f}")
        report.append(f"- **Specificity**: {metrics['specificity']:.3f}")
        report.append(f"- **Balanced Accuracy**: {metrics['balanced_accuracy']:.3f}\n")
        
        # Regression Metrics
        report.append("## Regression Metrics\n")
        report.append(f"- **Mean Squared Error**: {metrics['mse']:.3f}")
        report.append(f"- **Mean Absolute Error**: {metrics['mae']:.3f}")
        report.append(f"- **R² Score**: {metrics['r2_score']:.3f}")
        report.append(f"- **MAPE**: {metrics['mape']:.1f}%\n")
        
        # Loneliness-Specific Metrics
        report.append("## Loneliness Detection Metrics\n")
        report.append(f"- **Risk Level Accuracy**: {metrics['risk_level_accuracy']:.3f}")
        report.append(f"- **High-Risk Precision**: {metrics['high_risk_precision']:.3f}")
        report.append(f"- **High-Risk Recall**: {metrics['high_risk_recall']:.3f}")
        report.append(f"- **Early Detection Accuracy**: {metrics['early_detection_accuracy']:.3f}\n")
        
        # Interpretation
        report.append("## Model Interpretation\n")
        
        if metrics['f1_score'] >= 0.8:
            report.append("✅ **Excellent Performance**: Model shows strong ability to detect loneliness")
        elif metrics['f1_score'] >= 0.7:
            report.append("✅ **Good Performance**: Model performs well for loneliness detection")
        elif metrics['f1_score'] >= 0.6:
            report.append("⚠️ **Moderate Performance**: Model shows acceptable but improvable results")
        else:
            report.append("❌ **Poor Performance**: Model needs significant improvement")
        
        if metrics['high_risk_recall'] >= 0.8:
            report.append("✅ **Strong High-Risk Detection**: Effectively identifies severely lonely individuals")
        else:
            report.append("⚠️ **Needs Improvement**: High-risk detection could be enhanced")
        
        # Recommendations
        report.append("\n## Recommendations\n")
        
        if metrics['precision'] < 0.7:
            report.append("- Consider techniques to reduce false positives")
        
        if metrics['recall'] < 0.7:
            report.append("- Focus on improving sensitivity to detect more lonely individuals")
        
        if metrics['pearson_correlation'] < 0.6:
            report.append("- Investigate feature engineering to improve score prediction")
        
        return "\n".join(report)

class ExplainabilityEvaluator:
    """
    Evaluate the quality of model explanations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_explanation_quality(self, explanations: List[Dict]) -> Dict[str, float]:
        """Evaluate the quality of explanations"""
        
        metrics = {}
        
        # Consistency metrics
        metrics['attention_consistency'] = self._evaluate_attention_consistency(explanations)
        
        # Faithfulness metrics
        metrics['faithfulness_score'] = self._evaluate_faithfulness(explanations)
        
        # Comprehensibility metrics
        metrics['comprehensibility_score'] = self._evaluate_comprehensibility(explanations)
        
        # Coverage metrics
        metrics['feature_coverage'] = self._evaluate_feature_coverage(explanations)
        
        return metrics
    
    def _evaluate_attention_consistency(self, explanations: List[Dict]) -> float:
        """Evaluate consistency of attention patterns"""
        if not explanations:
            return 0.0
        
        attention_patterns = []
        for exp in explanations:
            if 'attention' in exp and 'text_attention' in exp['attention']:
                weights = exp['attention']['text_attention'].get('weights', [])
                if weights:
                    attention_patterns.append(weights)
        
        if len(attention_patterns) < 2:
            return 1.0  # Perfect consistency if only one or no patterns
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(attention_patterns)):
            for j in range(i+1, len(attention_patterns)):
                if len(attention_patterns[i]) == len(attention_patterns[j]):
                    corr = np.corrcoef(attention_patterns[i], attention_patterns[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _evaluate_faithfulness(self, explanations: List[Dict]) -> float:
        """Evaluate faithfulness of explanations to model behavior"""
        # Simplified faithfulness evaluation
        faithfulness_scores = []
        
        for exp in explanations:
            score = 0.0
            
            # Check if prediction confidence aligns with explanation clarity
            pred_confidence = exp.get('prediction', {}).get('confidence', 0)
            
            # High confidence should have clear explanations
            if pred_confidence > 0.8:
                if 'attention' in exp:
                    score += 0.3
                if 'shap' in exp and len(exp['shap'].get('speech_shap', {}).get('values', [])) > 0:
                    score += 0.3
                if 'lime' in exp and len(exp['lime'].get('lime_explanation', {}).get('words', [])) > 0:
                    score += 0.4
            else:
                score = 0.5  # Lower baseline for low confidence
            
            faithfulness_scores.append(min(score, 1.0))
        
        return np.mean(faithfulness_scores) if faithfulness_scores else 0.0
    
    def _evaluate_comprehensibility(self, explanations: List[Dict]) -> float:
        """Evaluate comprehensibility of explanations"""
        comprehensibility_scores = []
        
        for exp in explanations:
            score = 0.0
            
            # Natural language explanation should be present and meaningful
            natural_exp = exp.get('natural_language', '')
            if natural_exp and len(natural_exp.split()) > 10:
                score += 0.5
            
            # Top features should be interpretable
            if 'feature_importance' in exp:
                importance = exp['feature_importance']
                if 'top_speech_features' in importance and len(importance['top_speech_features']) > 0:
                    score += 0.25
                if 'top_text_patterns' in importance:
                    score += 0.25
            
            comprehensibility_scores.append(min(score, 1.0))
        
        return np.mean(comprehensibility_scores) if comprehensibility_scores else 0.0
    
    def _evaluate_feature_coverage(self, explanations: List[Dict]) -> float:
        """Evaluate coverage of different feature types in explanations"""
        coverage_scores = []
        
        for exp in explanations:
            covered_modalities = 0
            total_modalities = 2  # Speech and text
            
            # Check speech feature coverage
            if 'feature_importance' in exp and 'top_speech_features' in exp['feature_importance']:
                if len(exp['feature_importance']['top_speech_features']) > 0:
                    covered_modalities += 1
            
            # Check text feature coverage
            if 'attention' in exp and 'text_attention' in exp['attention']:
                if len(exp['attention']['text_attention'].get('tokens', [])) > 0:
                    covered_modalities += 1
            
            coverage_scores.append(covered_modalities / total_modalities)
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
