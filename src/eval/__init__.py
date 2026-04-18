"""Evaluation metrics and analysis for iris recognition."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

from ..features import IrisTemplate

logger = logging.getLogger(__name__)


class BiometricEvaluator:
    """Evaluator for biometric recognition systems."""
    
    def __init__(self):
        """Initialize the biometric evaluator."""
        self.scores = []
        self.labels = []
        self.identifiers = []
    
    def add_scores(
        self,
        scores: List[float],
        labels: List[bool],
        identifiers: Optional[List[str]] = None
    ) -> None:
        """Add similarity scores and labels for evaluation.
        
        Args:
            scores: List of similarity scores.
            labels: List of true labels (True for genuine, False for impostor).
            identifiers: Optional list of identifiers.
        """
        self.scores.extend(scores)
        self.labels.extend(labels)
        if identifiers:
            self.identifiers.extend(identifiers)
    
    def clear(self) -> None:
        """Clear all stored scores and labels."""
        self.scores = []
        self.labels = []
        self.identifiers = []
    
    def compute_eer(self) -> Tuple[float, float]:
        """Compute Equal Error Rate (EER).
        
        Returns:
            Tuple of (EER_threshold, EER_value).
        """
        if not self.scores or not self.labels:
            raise ValueError("No scores or labels available for EER computation")
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Sort scores
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Compute FAR and FRR for each threshold
        thresholds = sorted_scores
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            # False Accept Rate (FAR) - impostors accepted
            impostor_scores = scores[~labels]
            far = np.sum(impostor_scores >= threshold) / len(impostor_scores) if len(impostor_scores) > 0 else 0
            
            # False Reject Rate (FRR) - genuines rejected
            genuine_scores = scores[labels]
            frr = np.sum(genuine_scores < threshold) / len(genuine_scores) if len(genuine_scores) > 0 else 0
            
            far_values.append(far)
            frr_values.append(frr)
        
        far_values = np.array(far_values)
        frr_values = np.array(frr_values)
        
        # Find EER threshold (where FAR ≈ FRR)
        eer_idx = np.argmin(np.abs(far_values - frr_values))
        eer_threshold = thresholds[eer_idx]
        eer_value = (far_values[eer_idx] + frr_values[eer_idx]) / 2
        
        return eer_threshold, eer_value
    
    def compute_mindcf(
        self,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
        p_target: float = 0.5
    ) -> Tuple[float, float]:
        """Compute minimum Detection Cost Function (minDCF).
        
        Args:
            c_miss: Cost of miss (false reject).
            c_fa: Cost of false alarm (false accept).
            p_target: Prior probability of target.
            
        Returns:
            Tuple of (minDCF_threshold, minDCF_value).
        """
        if not self.scores or not self.labels:
            raise ValueError("No scores or labels available for minDCF computation")
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Sort scores
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Compute DCF for each threshold
        thresholds = sorted_scores
        dcf_values = []
        
        for threshold in thresholds:
            # False Accept Rate (FAR)
            impostor_scores = scores[~labels]
            far = np.sum(impostor_scores >= threshold) / len(impostor_scores) if len(impostor_scores) > 0 else 0
            
            # False Reject Rate (FRR)
            genuine_scores = scores[labels]
            frr = np.sum(genuine_scores < threshold) / len(genuine_scores) if len(genuine_scores) > 0 else 0
            
            # Detection Cost Function
            dcf = c_miss * frr * p_target + c_fa * far * (1 - p_target)
            dcf_values.append(dcf)
        
        dcf_values = np.array(dcf_values)
        
        # Find minimum DCF
        min_dcf_idx = np.argmin(dcf_values)
        min_dcf_threshold = thresholds[min_dcf_idx]
        min_dcf_value = dcf_values[min_dcf_idx]
        
        return min_dcf_threshold, min_dcf_value
    
    def compute_roc_metrics(self) -> Dict[str, float]:
        """Compute ROC curve metrics.
        
        Returns:
            Dictionary containing ROC metrics.
        """
        if not self.scores or not self.labels:
            raise ValueError("No scores or labels available for ROC computation")
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'auc': auc,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx],
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def compute_precision_recall_metrics(self) -> Dict[str, float]:
        """Compute precision-recall metrics.
        
        Returns:
            Dictionary containing precision-recall metrics.
        """
        if not self.scores or not self.labels:
            raise ValueError("No scores or labels available for precision-recall computation")
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        avg_precision = average_precision_score(labels, scores)
        
        # Find optimal threshold (F1 score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        
        return {
            'average_precision': avg_precision,
            'optimal_threshold': optimal_threshold,
            'optimal_precision': precision[optimal_idx],
            'optimal_recall': recall[optimal_idx],
            'optimal_f1': f1_scores[optimal_idx],
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
    
    def compute_confusion_matrix(self, threshold: float) -> np.ndarray:
        """Compute confusion matrix for a given threshold.
        
        Args:
            threshold: Decision threshold.
            
        Returns:
            Confusion matrix.
        """
        if not self.scores or not self.labels:
            raise ValueError("No scores or labels available for confusion matrix computation")
        
        scores = np.array(self.scores)
        labels = np.array(self.labels)
        
        # Predictions based on threshold
        predictions = scores >= threshold
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        return cm
    
    def generate_report(self, threshold: Optional[float] = None) -> Dict[str, any]:
        """Generate comprehensive evaluation report.
        
        Args:
            threshold: Decision threshold. If None, uses EER threshold.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if not self.scores or not self.labels:
            raise ValueError("No scores or labels available for report generation")
        
        # Compute EER
        eer_threshold, eer_value = self.compute_eer()
        
        # Use provided threshold or EER threshold
        if threshold is None:
            threshold = eer_threshold
        
        # Compute all metrics
        roc_metrics = self.compute_roc_metrics()
        pr_metrics = self.compute_precision_recall_metrics()
        mindcf_threshold, mindcf_value = self.compute_mindcf()
        cm = self.compute_confusion_matrix(threshold)
        
        # Compute additional metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # FAR and FRR
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        report = {
            'threshold': threshold,
            'eer': {
                'threshold': eer_threshold,
                'value': eer_value
            },
            'mindcf': {
                'threshold': mindcf_threshold,
                'value': mindcf_value
            },
            'roc': roc_metrics,
            'precision_recall': pr_metrics,
            'confusion_matrix': cm,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1_score,
                'far': far,
                'frr': frr
            },
            'sample_counts': {
                'total': len(self.scores),
                'genuine': np.sum(self.labels),
                'impostor': np.sum(~np.array(self.labels))
            }
        }
        
        return report


class IrisVisualizer:
    """Visualization tools for iris recognition evaluation."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer.
        
        Args:
            figsize: Default figure size.
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_roc_curve(
        self,
        evaluator: BiometricEvaluator,
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve.
        
        Args:
            evaluator: BiometricEvaluator instance.
            save_path: Optional path to save the plot.
        """
        roc_metrics = evaluator.compute_roc_metrics()
        
        plt.figure(figsize=self.figsize)
        plt.plot(roc_metrics['fpr'], roc_metrics['tpr'], 
                label=f'ROC Curve (AUC = {roc_metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        # Mark optimal point
        plt.plot(roc_metrics['optimal_fpr'], roc_metrics['optimal_tpr'], 
                'ro', label=f'Optimal Point (FPR={roc_metrics["optimal_fpr"]:.3f}, TPR={roc_metrics["optimal_tpr"]:.3f})')
        
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve for Iris Recognition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        evaluator: BiometricEvaluator,
        save_path: Optional[str] = None
    ) -> None:
        """Plot precision-recall curve.
        
        Args:
            evaluator: BiometricEvaluator instance.
            save_path: Optional path to save the plot.
        """
        pr_metrics = evaluator.compute_precision_recall_metrics()
        
        plt.figure(figsize=self.figsize)
        plt.plot(pr_metrics['recall'], pr_metrics['precision'], 
                label=f'PR Curve (AP = {pr_metrics["average_precision"]:.3f})')
        
        # Mark optimal point
        plt.plot(pr_metrics['optimal_recall'], pr_metrics['optimal_precision'], 
                'ro', label=f'Optimal Point (R={pr_metrics["optimal_recall"]:.3f}, P={pr_metrics["optimal_precision"]:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Iris Recognition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_det_curve(
        self,
        evaluator: BiometricEvaluator,
        save_path: Optional[str] = None
    ) -> None:
        """Plot DET (Detection Error Tradeoff) curve.
        
        Args:
            evaluator: BiometricEvaluator instance.
            save_path: Optional path to save the plot.
        """
        scores = np.array(evaluator.scores)
        labels = np.array(evaluator.labels)
        
        # Compute FAR and FRR for different thresholds
        thresholds = np.linspace(min(scores), max(scores), 1000)
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            # False Accept Rate (FAR)
            impostor_scores = scores[~labels]
            far = np.sum(impostor_scores >= threshold) / len(impostor_scores) if len(impostor_scores) > 0 else 0
            
            # False Reject Rate (FRR)
            genuine_scores = scores[labels]
            frr = np.sum(genuine_scores < threshold) / len(genuine_scores) if len(genuine_scores) > 0 else 0
            
            far_values.append(far)
            frr_values.append(frr)
        
        # Plot DET curve
        plt.figure(figsize=self.figsize)
        plt.semilogx(far_values, frr_values, label='DET Curve')
        
        # Mark EER point
        eer_threshold, eer_value = evaluator.compute_eer()
        plt.semilogx([eer_value], [eer_value], 'ro', label=f'EER = {eer_value:.3f}')
        
        plt.xlabel('False Accept Rate (FAR)')
        plt.ylabel('False Reject Rate (FRR)')
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_distributions(
        self,
        evaluator: BiometricEvaluator,
        save_path: Optional[str] = None
    ) -> None:
        """Plot score distributions for genuine and impostor samples.
        
        Args:
            evaluator: BiometricEvaluator instance.
            save_path: Optional path to save the plot.
        """
        scores = np.array(evaluator.scores)
        labels = np.array(evaluator.labels)
        
        genuine_scores = scores[labels]
        impostor_scores = scores[~labels]
        
        plt.figure(figsize=self.figsize)
        
        # Plot histograms
        plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', density=True)
        plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', density=True)
        
        # Mark EER threshold
        eer_threshold, eer_value = evaluator.compute_eer()
        plt.axvline(eer_threshold, color='red', linestyle='--', 
                   label=f'EER Threshold = {eer_threshold:.3f}')
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Score Distributions for Genuine and Impostor Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(
        self,
        evaluator: BiometricEvaluator,
        threshold: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            evaluator: BiometricEvaluator instance.
            threshold: Decision threshold. If None, uses EER threshold.
            save_path: Optional path to save the plot.
        """
        if threshold is None:
            threshold, _ = evaluator.compute_eer()
        
        cm = evaluator.compute_confusion_matrix(threshold)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Reject', 'Accept'],
                   yticklabels=['Impostor', 'Genuine'])
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Threshold = {threshold:.3f})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_evaluation_summary(
        self,
        evaluator: BiometricEvaluator,
        save_path: Optional[str] = None
    ) -> None:
        """Plot comprehensive evaluation summary.
        
        Args:
            evaluator: BiometricEvaluator instance.
            save_path: Optional path to save the plot.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ROC Curve
        roc_metrics = evaluator.compute_roc_metrics()
        axes[0, 0].plot(roc_metrics['fpr'], roc_metrics['tpr'], 
                       label=f'AUC = {roc_metrics["auc"]:.3f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        pr_metrics = evaluator.compute_precision_recall_metrics()
        axes[0, 1].plot(pr_metrics['recall'], pr_metrics['precision'],
                       label=f'AP = {pr_metrics["average_precision"]:.3f}')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Score Distributions
        scores = np.array(evaluator.scores)
        labels = np.array(evaluator.labels)
        genuine_scores = scores[labels]
        impostor_scores = scores[~labels]
        
        axes[1, 0].hist(genuine_scores, bins=30, alpha=0.7, label='Genuine', density=True)
        axes[1, 0].hist(impostor_scores, bins=30, alpha=0.7, label='Impostor', density=True)
        eer_threshold, _ = evaluator.compute_eer()
        axes[1, 0].axvline(eer_threshold, color='red', linestyle='--', 
                          label=f'EER = {eer_threshold:.3f}')
        axes[1, 0].set_xlabel('Similarity Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Score Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = evaluator.compute_confusion_matrix(eer_threshold)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Reject', 'Accept'],
                   yticklabels=['Impostor', 'Genuine'], ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title(f'Confusion Matrix (EER Threshold)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
