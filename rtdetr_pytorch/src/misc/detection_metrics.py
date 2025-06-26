

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import torchvision.ops as ops
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize


class DetectionMetricsCalculator:
    """detection metrics calculator, calculate various detection performance metrics"""
    
    def __init__(self, num_classes: int, device: torch.device = None):
        self.num_classes = num_classes
        self.device = device or torch.device('cpu')
        self.reset()
    
    def reset(self):
        """reset all metrics"""
        self.all_predictions = []
        self.all_targets = []
        self.all_scores = []
        self.all_boxes = []
        self.all_target_boxes = []
        self.all_iou_scores = []
        self.all_nms_results = []
        self.all_labels = []
    
    def update(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], 
               postprocessor=None, orig_target_sizes=None):
        """update metrics calculation"""
        # get prediction results
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # calculate classification prediction
        if postprocessor and postprocessor.use_focal_loss:
            pred_scores = torch.sigmoid(pred_logits)
            pred_labels = torch.argmax(pred_scores, dim=-1)
        else:
            pred_scores = F.softmax(pred_logits, dim=-1)[:, :, :-1]
            pred_labels = torch.argmax(pred_scores, dim=-1)
        
        # process targets
        batch_size = len(targets)
        for i in range(batch_size):
            target = targets[i]
            target_labels = target['labels']
            target_boxes = target['boxes']
            
            # get current sample's prediction
            sample_pred_labels = pred_labels[i]
            sample_pred_scores = pred_scores[i]
            sample_pred_boxes = pred_boxes[i]
            
            # filter background class (usually the last class)
            valid_pred_mask = sample_pred_labels < self.num_classes
            if valid_pred_mask.sum() > 0:
                valid_pred_labels = sample_pred_labels[valid_pred_mask]
                valid_pred_scores = sample_pred_scores[valid_pred_mask]
                valid_pred_boxes = sample_pred_boxes[valid_pred_mask]
                
                # calculate IoU
                if len(target_boxes) > 0 and len(valid_pred_boxes) > 0:
                    iou_matrix = self._calculate_iou_matrix(valid_pred_boxes, target_boxes)
                    max_ious, max_indices = torch.max(iou_matrix, dim=1)
                    
                    # match predictions and targets
                    matched_targets = []
                    for j, max_iou in enumerate(max_ious):
                        if max_iou > 0.5:  # IoU threshold
                            matched_targets.append(target_labels[max_indices[j]])
                        else:
                            matched_targets.append(-1)  # not matched
                    
                    matched_targets = torch.tensor(matched_targets, device=self.device)
                    
                    # collect all prediction scores and labels
                    if valid_pred_scores.numel() > 0:
                        self.all_scores.extend(valid_pred_scores.max(dim=-1)[0].detach().cpu().numpy())
                        self.all_labels.extend(valid_pred_labels.detach().cpu().numpy())
                    self.all_predictions.extend(valid_pred_labels.detach().cpu().numpy())
                    self.all_targets.extend(matched_targets.detach().cpu().numpy())
                    self.all_boxes.extend(valid_pred_boxes.detach().cpu().numpy())
                    self.all_target_boxes.extend(target_boxes.detach().cpu().numpy())
                    self.all_iou_scores.extend(max_ious.detach().cpu().numpy())
    
    def _calculate_iou_matrix(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """calculate IoU matrix between predicted and target boxes"""
        # convert to xyxy format
        if pred_boxes.shape[-1] == 4:
            pred_boxes_xyxy = pred_boxes
            target_boxes_xyxy = target_boxes
        else:
            # 假设是cxcywh格式
            pred_boxes_xyxy = ops.box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy')
            target_boxes_xyxy = ops.box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        
        # 计算IoU矩阵
        iou_matrix = ops.box_iou(pred_boxes_xyxy, target_boxes_xyxy)
        return iou_matrix
    
    def compute_accuracy(self) -> float:
        """calculate classification accuracy"""
        if not self.all_predictions:
            return 0.0
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # 只考虑有效的目标（非-1）
        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            return 0.0
        
        correct = (predictions[valid_mask] == targets[valid_mask]).sum()
        total = valid_mask.sum()
        return correct / total if total > 0 else 0.0
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """calculate confusion matrix"""
        if not self.all_predictions:
            return np.zeros((self.num_classes, self.num_classes))
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # only consider valid targets
        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(targets[valid_mask], predictions[valid_mask], 
                              labels=range(self.num_classes))
    
    def compute_precision_recall(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """calculate precision and recall for each class"""
        if not self.all_predictions:
            return {}, {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        scores = np.array(self.all_scores)
        
        precision_per_class = {}
        recall_per_class = {}
        
        for class_id in range(self.num_classes):
            # binary classification: current class vs other classes
            binary_targets = (targets == class_id).astype(int)
            binary_predictions = (predictions == class_id).astype(int)
            
            # only consider valid targets
            valid_mask = targets != -1
            if valid_mask.sum() == 0:
                precision_per_class[class_id] = 0.0
                recall_per_class[class_id] = 0.0
                continue
            
            binary_targets = binary_targets[valid_mask]
            binary_predictions = binary_predictions[valid_mask]
            
            # 计算TP, FP, FN
            tp = ((binary_predictions == 1) & (binary_targets == 1)).sum()
            fp = ((binary_predictions == 1) & (binary_targets == 0)).sum()
            fn = ((binary_predictions == 0) & (binary_targets == 1)).sum()
            
            # 计算精确率和召回率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_per_class[class_id] = precision
            recall_per_class[class_id] = recall
        
        return precision_per_class, recall_per_class
    
    def compute_mean_iou(self) -> float:
        """calculate mean IoU"""
        if not self.all_iou_scores:
            return 0.0
        
        return np.mean(self.all_iou_scores)
    
    def compute_roc_auc(self) -> Dict[int, float]:
        """calculate ROC AUC for each class"""
        if not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        scores = np.array(self.all_scores)
        
        auc_per_class = {}
        
        for class_id in range(self.num_classes):
            # binary classification: current class vs other classes
            binary_targets = (targets == class_id).astype(int)
            
            # only consider valid targets
            valid_mask = targets != -1
            if valid_mask.sum() == 0:
                auc_per_class[class_id] = 0.0
                continue
            
            binary_targets = binary_targets[valid_mask]
            binary_scores = scores[valid_mask]
            
            # 计算ROC AUC
            try:
                fpr, tpr, _ = roc_curve(binary_targets, binary_scores)
                roc_auc = auc(fpr, tpr)
                auc_per_class[class_id] = roc_auc
            except:
                auc_per_class[class_id] = 0.0
        
        return auc_per_class
    
    def compute_nms_metrics(self, iou_threshold: float = 0.5) -> Dict[str, float]:
        """calculate NMS related metrics"""
        if not self.all_boxes:
            return {'nms_boxes_removed': 0, 'nms_avg_boxes_per_image': 0}
        
        boxes = np.array(self.all_boxes)
        scores = np.array(self.all_scores)
        labels = np.array(self.all_predictions)
        
        # group by class and perform NMS
        nms_boxes_removed = 0
        total_boxes_before_nms = len(boxes)
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            if label_mask.sum() > 1:
                label_boxes = boxes[label_mask]
                label_scores = scores[label_mask]
                
                keep_indices = ops.nms(
                    torch.tensor(label_boxes), 
                    torch.tensor(label_scores), 
                    iou_threshold
                )
                
                nms_boxes_removed += label_mask.sum() - len(keep_indices)
        
        return {
            'nms_boxes_removed': nms_boxes_removed,
            'nms_avg_boxes_per_image': total_boxes_before_nms / max(1, len(np.unique(labels)))
        }
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """calculate all metrics"""
        metrics = {}
        
        # basic classification metrics
        metrics['accuracy'] = self.compute_accuracy()
        metrics['confusion_matrix'] = self.compute_confusion_matrix()
        
        # precision and recall
        precision_per_class, recall_per_class = self.compute_precision_recall()
        metrics['precision_per_class'] = precision_per_class
        metrics['recall_per_class'] = recall_per_class
        metrics['mean_precision'] = np.mean(list(precision_per_class.values()))
        metrics['mean_recall'] = np.mean(list(recall_per_class.values()))
        
        # IoU metrics
        metrics['mean_iou'] = self.compute_mean_iou()
        
        # ROC AUC metrics
        auc_per_class = self.compute_roc_auc()
        metrics['roc_auc_per_class'] = auc_per_class
        metrics['mean_roc_auc'] = np.mean(list(auc_per_class.values()))
        
        # NMS related metrics
        nms_removed = 0
        for boxes, scores in zip(self.all_boxes, self.all_scores):
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
            scores_tensor = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
            # check shape
            if boxes_tensor.numel() == 0 or scores_tensor.numel() == 0:
                continue
            if boxes_tensor.dim() == 1:
                if boxes_tensor.shape[0] == 4:
                    boxes_tensor = boxes_tensor.unsqueeze(0)
                else:
                    continue
            if boxes_tensor.shape[1] != 4:
                continue
            if scores_tensor.dim() != 1 or scores_tensor.shape[0] != boxes_tensor.shape[0]:
                continue
            keep_indices = ops.nms(
                boxes_tensor,
                scores_tensor,
                iou_threshold=0.5
            )
            nms_removed += boxes_tensor.shape[0] - keep_indices.shape[0]
        metrics['nms_boxes_removed'] = nms_removed
        
        return metrics


class StepMetricsTracker:
    """step-level metrics tracker"""
    
    def __init__(self, num_classes: int, device: torch.device = None):
        self.num_classes = num_classes
        self.device = device or torch.device('cpu')
        self.metrics_calculator = DetectionMetricsCalculator(num_classes, device)
        self.step_metrics = []
        self.record_interval = 50  # 每50步记录一次
    
    def update(self, step: int, outputs: Dict[str, torch.Tensor], 
               targets: List[Dict[str, torch.Tensor]], 
               postprocessor=None, orig_target_sizes=None):
        """update step-level metrics"""
        self.metrics_calculator.update(outputs, targets, postprocessor, orig_target_sizes)
        
        # record metrics every 50 steps
        if step % self.record_interval == 0:
            metrics = self.metrics_calculator.compute_all_metrics()
            metrics['step'] = step
            self.step_metrics.append(metrics)
            
            # 重置计算器以开始新的统计
            self.metrics_calculator.reset()
    
    def get_step_metrics(self) -> List[Dict[str, Any]]:
        """get all step-level metrics"""
        return self.step_metrics
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """get latest metrics"""
        if self.step_metrics:
            return self.step_metrics[-1]
        return None


def calculate_detection_metrics(outputs: Dict[str, torch.Tensor], 
                              targets: List[Dict[str, torch.Tensor]], 
                              num_classes: int,
                              postprocessor=None,
                              orig_target_sizes=None,
                              device: torch.device = None) -> Dict[str, Any]:
    """convenient function: calculate detection metrics"""
    calculator = DetectionMetricsCalculator(num_classes, device)
    calculator.update(outputs, targets, postprocessor, orig_target_sizes)
    return calculator.compute_all_metrics() 