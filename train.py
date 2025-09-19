"""
Main Training Script
Complete training pipeline for loneliness detection model
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models import LonelinessDetectionModel, ModelTrainer, LonelinessDataset
from src.utils import Config, Logger, DataPipeline, ModelEvaluator
from src.explainability import ModelExplainer
from transformers import AutoTokenizer

def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Loneliness Detection Model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-config', type=str, required=True,
                       help='Data configuration (synthetic/csv/directory)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for models and results')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger('training', os.path.join(args.output_dir, 'training.log'))
    log = logger.get_logger()
    
    log.info("Starting loneliness detection model training...")
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    log.info(f"Using device: {device}")
    
    # Data pipeline configuration
    pipeline_config = {
        'data_config': {'synthetic': {'n_samples': 1000}},  # Default synthetic data
        'batch_size': config.get('training.batch_size', 32),
        'test_size': config.get('data.test_size', 0.2),
        'val_size': config.get('data.val_size', 0.1),
        'cache_dir': os.path.join(args.output_dir, 'cache'),
        'apply_augmentation': True
    }
    
    # Override with user-specified data config
    if args.data_config == 'synthetic':
        pipeline_config['data_config'] = {'synthetic': {'n_samples': 1000}}
    elif args.data_config.endswith('.csv'):
        pipeline_config['data_config'] = {'csv_file': args.data_config}
    else:
        pipeline_config['data_config'] = {'directory': args.data_config}
    
    # Initialize data pipeline
    log.info("Setting up data pipeline...")
    data_pipeline = DataPipeline(pipeline_config)
    dataloaders = data_pipeline.run_pipeline()
    
    # Initialize model
    log.info("Initializing model...")
    model = LonelinessDetectionModel(
        speech_input_dim=config.get('model.speech_input_dim', 128),
        text_model_name=config.get('model.text_model_name', 'bert-base-uncased'),
        hidden_dim=config.get('model.hidden_dim', 256),
        dropout=config.get('model.dropout', 0.3)
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.get('model.text_model_name', 'bert-base-uncased'))
    
    # Update datasets with tokenizer
    for split_name, dataloader in dataloaders.items():
        dataloader.dataset.tokenizer = tokenizer
    
    # Initialize trainer
    log.info("Setting up trainer...")
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=config.get('training.learning_rate', 2e-5),
        weight_decay=config.get('training.weight_decay', 0.01)
    )
    
    # Train model
    log.info("Starting training...")
    model_save_path = os.path.join(args.output_dir, 'best_model.pth')
    
    training_history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=config.get('training.num_epochs', 50),
        early_stopping_patience=config.get('training.early_stopping_patience', 10),
        save_path=model_save_path
    )
    
    # Evaluate on test set
    log.info("Evaluating on test set...")
    test_metrics = trainer.validate_epoch(dataloaders['test'])
    
    # Generate predictions for detailed evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in dataloaders['test']:
            speech_features = batch['speech_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            output = model(speech_features, input_ids, attention_mask)
            
            test_predictions.extend(output['predictions'].cpu().numpy())
            test_targets.extend(labels.numpy())
    
    # Comprehensive evaluation
    evaluator = ModelEvaluator()
    detailed_metrics = evaluator.evaluate_predictions(test_targets, test_predictions)
    
    # Generate evaluation report
    evaluation_report = evaluator.create_evaluation_report(
        test_targets, test_predictions, "Loneliness Detection Model"
    )
    
    # Save results
    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'detailed_metrics': detailed_metrics,
        'model_config': {
            'speech_input_dim': config.get('model.speech_input_dim', 128),
            'text_model_name': config.get('model.text_model_name', 'bert-base-uncased'),
            'hidden_dim': config.get('model.hidden_dim', 256),
            'dropout': config.get('model.dropout', 0.3)
        },
        'training_config': {
            'learning_rate': config.get('training.learning_rate', 2e-5),
            'batch_size': config.get('training.batch_size', 32),
            'num_epochs': config.get('training.num_epochs', 50)
        }
    }
    
    # Save evaluation report
    report_path = os.path.join(args.output_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(evaluation_report)
    
    # Save results
    import json
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model configuration
    config_save_path = os.path.join(args.output_dir, 'model_config.yaml')
    config.save_config(config_save_path)
    
    # Test explainability
    log.info("Testing explainability framework...")
    explainer = ModelExplainer(model, tokenizer, device)
    
    # Get a sample from test set for explanation
    sample_batch = next(iter(dataloaders['test']))
    sample_speech = sample_batch['speech_features'][0].numpy()
    sample_text = "I feel so alone these days. Nobody visits me anymore."
    
    explanation = explainer.explain_prediction(
        sample_speech, sample_text, ['attention', 'shap']
    )
    
    # Save sample explanation
    explanation_path = os.path.join(args.output_dir, 'sample_explanation.json')
    with open(explanation_path, 'w') as f:
        json.dump(explanation, f, indent=2, default=str)
    
    # Print summary
    log.info("Training completed successfully!")
    log.info(f"Final test accuracy: {detailed_metrics['accuracy']:.3f}")
    log.info(f"Final test F1-score: {detailed_metrics['f1_score']:.3f}")
    log.info(f"Final test AUC: {detailed_metrics['auc_roc']:.3f}")
    log.info(f"Results saved to: {args.output_dir}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"📊 Test Accuracy: {detailed_metrics['accuracy']:.3f}")
    print(f"📊 Test F1-Score: {detailed_metrics['f1_score']:.3f}")
    print(f"📊 Test AUC: {detailed_metrics['auc_roc']:.3f}")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"🤖 Model saved to: {model_save_path}")
    print("="*50)

if __name__ == "__main__":
    main()
