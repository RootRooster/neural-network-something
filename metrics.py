import torch
import torchmetrics
from torchmetrics import MetricCollection

def setup_metrics(device, num_classes=13, threshold=0.5):
    """Setup all metrics using torchmetrics"""
    
    # Binary classification metrics (aneurysm presence)
    binary_metrics = MetricCollection({
        'binary_accuracy': torchmetrics.Accuracy(task='binary', threshold=threshold),
        'binary_recall': torchmetrics.Recall(task='binary', threshold=threshold),
        'binary_precision': torchmetrics.Precision(task='binary', threshold=threshold),
        'binary_f1': torchmetrics.F1Score(task='binary', threshold=threshold),
    }).to(device)
    
    # Multi-label classification metrics (aneurysm locations)
    multilabel_metrics = MetricCollection({
        'exact_match_accuracy': torchmetrics.ExactMatch(task='multilabel', num_labels=num_classes, threshold=threshold),
        'macro_recall': torchmetrics.Recall(task='multilabel', num_labels=num_classes, threshold=threshold, average='macro'),
        'micro_recall': torchmetrics.Recall(task='multilabel', num_labels=num_classes, threshold=threshold, average='micro'),
        'macro_precision': torchmetrics.Precision(task='multilabel', num_labels=num_classes, threshold=threshold, average='macro'),
        'micro_precision': torchmetrics.Precision(task='multilabel', num_labels=num_classes, threshold=threshold, average='micro'),
        'macro_f1': torchmetrics.F1Score(task='multilabel', num_labels=num_classes, threshold=threshold, average='macro'),
        'micro_f1': torchmetrics.F1Score(task='multilabel', num_labels=num_classes, threshold=threshold, average='micro'),
        'hamming_distance': torchmetrics.HammingDistance(task='multilabel', num_labels=num_classes, threshold=threshold),
    }).to(device)
    
    # Per-class metrics for detailed analysis
    per_class_metrics = MetricCollection({
        'per_class_accuracy': torchmetrics.Accuracy(task='multilabel', num_labels=num_classes, threshold=threshold, average=None),
        'per_class_recall': torchmetrics.Recall(task='multilabel', num_labels=num_classes, threshold=threshold, average=None),
        'per_class_precision': torchmetrics.Precision(task='multilabel', num_labels=num_classes, threshold=threshold, average=None),
        'per_class_f1': torchmetrics.F1Score(task='multilabel', num_labels=num_classes, threshold=threshold, average=None),
    }).to(device)
    
    return binary_metrics, multilabel_metrics, per_class_metrics

def extract_metrics_results(binary_results, multilabel_results, per_class_results):
    """Extract and organize metrics results"""
    
    metrics = {}
    
    # Binary metrics
    metrics['binary_accuracy'] = binary_results['binary_accuracy'].item()
    metrics['binary_recall'] = binary_results['binary_recall'].item()
    metrics['binary_precision'] = binary_results['binary_precision'].item()
    metrics['binary_f1'] = binary_results['binary_f1'].item()
    
    # Multi-label metrics
    metrics['exact_match_accuracy'] = multilabel_results['exact_match_accuracy'].item()
    metrics['macro_recall'] = multilabel_results['macro_recall'].item()
    metrics['micro_recall'] = multilabel_results['micro_recall'].item()
    metrics['macro_precision'] = multilabel_results['macro_precision'].item()
    metrics['micro_precision'] = multilabel_results['micro_precision'].item()
    metrics['macro_f1'] = multilabel_results['macro_f1'].item()
    metrics['micro_f1'] = multilabel_results['micro_f1'].item()
    metrics['hamming_distance'] = multilabel_results['hamming_distance'].item()
    
    # Per-class metrics (returns as tensors)
    metrics['per_class_accuracy'] = per_class_results['per_class_accuracy'].cpu().numpy().tolist()
    metrics['per_class_recall'] = per_class_results['per_class_recall'].cpu().numpy().tolist()
    metrics['per_class_precision'] = per_class_results['per_class_precision'].cpu().numpy().tolist()
    metrics['per_class_f1'] = per_class_results['per_class_f1'].cpu().numpy().tolist()
    
    # Compute mean class metrics
    metrics['mean_class_accuracy'] = sum(metrics['per_class_accuracy']) / len(metrics['per_class_accuracy'])
    metrics['mean_class_recall'] = sum(metrics['per_class_recall']) / len(metrics['per_class_recall'])
    metrics['mean_class_precision'] = sum(metrics['per_class_precision']) / len(metrics['per_class_precision'])
    metrics['mean_class_f1'] = sum(metrics['per_class_f1']) / len(metrics['per_class_f1'])
    
    return metrics