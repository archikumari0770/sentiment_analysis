import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import config
from models.fusion_model import MultimodalFusionModel
from utils.data_loader import get_data_loader

def evaluate_model(model, test_loader, device):
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for batch in test_loader:
            visual = batch['visual'].to(device)
            audio = batch['audio'].squeeze(1).to(device)
            labels = batch['label'].to(device)
            filenames = batch['filename']
            
            # Skip dummy labels
            if (labels == -1).any():
                continue
            
            outputs = model(visual, audio)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)
    
    return np.array(all_predictions), np.array(all_labels), all_filenames

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=config.SENTIMENT_LABELS.values(),
        yticklabels=config.SENTIMENT_LABELS.values()
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion matrix saved to {save_path}')

def main():
    device = config.DEVICE
    print(f'Using device: {device}')
    
    # Load test data
    test_loader = get_data_loader(
        video_dir=config.RAW_VIDEO_DIR,
        annotation_file=os.path.join(config.DATA_DIR, 'test_annotations.csv'),
        mode='test',
        shuffle=False
    )
    
    # Load model
    model = MultimodalFusionModel(num_classes=config.NUM_CLASSES).to(device)
    model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print('Error: Model not found. Please train the model first.')
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Evaluate
    print('\\nEvaluating model...')
    predictions, labels, filenames = evaluate_model(model, test_loader, device)
    
    # Classification report
    print('\\n' + '='*60)
    print('CLASSIFICATION REPORT')
    print('='*60)
    report = classification_report(
        labels, 
        predictions,
        target_names=list(config.SENTIMENT_LABELS.values()),
        digits=4
    )
    print(report)
    
    # Save report
    report_path = os.path.join(config.MODEL_SAVE_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'\\nReport saved to {report_path}')
    
    # Confusion matrix
    cm_path = os.path.join(config.MODEL_SAVE_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(labels, predictions, cm_path)
    
    # Per-file results
    results_path = os.path.join(config.MODEL_SAVE_DIR, 'per_file_results.txt')
    with open(results_path, 'w') as f:
        f.write('Filename,True Label,Predicted Label,Correct\\n')
        for filename, true_label, pred_label in zip(filenames, labels, predictions):
            correct = '✓' if true_label == pred_label else '✗'
            f.write(f'{filename},{config.SENTIMENT_LABELS[true_label]},'
                   f'{config.SENTIMENT_LABELS[pred_label]},{correct}\\n')
    print(f'Per-file results saved to {results_path}')

if __name__ == '__main__':
    main()