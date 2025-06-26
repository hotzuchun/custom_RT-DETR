import os
import json
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.distributed as tdist
from pycocotools.cocoeval import COCOeval
import contextlib
import gc

from .dist import is_dist_available_and_initialized, get_world_size, is_main_process


class YOLOResultsProcessor:
    
    def __init__(self, save_dir: str, experiment_name: str = "rtdetr_experiment", 
                 num_classes: int = 80, log_interval: int = 50, max_step_history: int = 1000):
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.num_classes = num_classes
        self.log_interval = log_interval
        self.max_step_history = max_step_history
        
        self.logs_dir = self.save_dir / "logs"
        self.plots_dir = self.save_dir / "plots"
        self.results_dir = self.save_dir / "results"
        
        for dir_path in [self.logs_dir, self.plots_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.epochs = []
        self.start_time = time.time()
        
        self.current_epoch = 0
        self.current_step = 0
        self.step_metrics = defaultdict(lambda: deque(maxlen=max_step_history))
        
        self.epoch_times = []
        self.step_times = deque(maxlen=100)
        
        self.best_map50 = 0.0
        self.best_map = 0.0
        self.best_epoch = 0
        
        self._init_log_files()
        
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _init_log_files(self):
        if not is_main_process():
            return
            
        self.train_log_file = self.logs_dir / "train_log.txt"
        self.results_file = self.results_dir / "results.csv"
        self.summary_file = self.logs_dir / "training_summary.txt"
        self.model_info_file = self.logs_dir / "model_info.txt"
        
        with open(self.train_log_file, 'w', encoding='utf-8') as f:
            f.write(f"RT-DETR Training Log (YOLO Style)\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
        
        self._init_csv_file()
    
    def _init_csv_file(self):
        if not is_main_process():
            return
            
        csv_headers = [
            'epoch', 'train/loss', 'train/loss_ce', 'train/loss_bbox', 'train/loss_giou',
            'val/mAP_0.5', 'val/mAP_0.5:0.95', 'val/recall',
            'lr/pg0', 'time/train_epoch', 'time/val_epoch'
        ]
        
        df = pd.DataFrame(columns=csv_headers)
        df.to_csv(self.results_file, index=False)
    
    def log_train_step(self, epoch: int, step: int, metrics: Dict[str, float], 
                      lr: float, batch_size: int, time_per_step: float):
        self.current_epoch = epoch
        self.current_step = step
        
        self.step_times.append(time_per_step)
        
        step_data = {
            'epoch': epoch,
            'step': step,
            'lr': lr,
            'batch_size': batch_size,
            'time_per_step': time_per_step,
            **metrics
        }
        
        for key, value in step_data.items():
            self.step_metrics[key].append(value)
        
        if step % self.log_interval == 0 and is_main_process():
            self._write_step_log(step_data)
    
    def log_train_epoch(self, epoch: int, metrics: Dict[str, float], 
                       total_time: float, samples_per_sec: float):
        self.epochs.append(epoch)
        self.epoch_times.append(total_time)
        
        epoch_data = {
            'epoch': epoch,
            'total_time': total_time,
            'samples_per_sec': samples_per_sec,
            **metrics
        }
        
        for key, value in epoch_data.items():
            self.train_metrics[key].append(value)
        
        if is_main_process():
            self._write_epoch_log('train', epoch_data)
            self._update_csv_file(epoch_data, 'train')
        
        self._cleanup_memory()
    
    def log_val_epoch(self, epoch: int, metrics: Dict[str, float], 
                     total_time: float, samples_per_sec: float):
        epoch_data = {
            'epoch': epoch,
            'total_time': total_time,
            'samples_per_sec': samples_per_sec,
            **metrics
        }
        
        for key, value in epoch_data.items():
            self.val_metrics[key].append(value)
        
        if 'coco_eval_bbox' in metrics:
            coco_stats = metrics['coco_eval_bbox']
            if len(coco_stats) >= 12:
                map50 = coco_stats[1]
                map = coco_stats[0]
                
                if map50 > self.best_map50:
                    self.best_map50 = map50
                    self.best_epoch = epoch
                
                if map > self.best_map:
                    self.best_map = map
        
        if is_main_process():
            self._write_epoch_log('val', epoch_data)
            self._update_csv_file(epoch_data, 'val')
        
        self._cleanup_memory()
    
    def _cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _write_step_log(self, step_data: Dict[str, Any]):
        with open(self.train_log_file, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {step_data['epoch']:3d} | Step {step_data['step']:6d} | ")
            f.write(f"LR: {step_data['lr']:.6f} | ")
            f.write(f"Loss: {step_data.get('loss', 0):.4f} | ")
            f.write(f"Time: {step_data['time_per_step']:.3f}s | ")
            f.write(f"Batch: {step_data['batch_size']}\n")
    
    def _write_epoch_log(self, phase: str, epoch_data: Dict[str, Any]):
        with open(self.train_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"{phase.upper()} Epoch {epoch_data['epoch']:3d}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Time: {epoch_data['total_time']:.2f}s | ")
            f.write(f"Speed: {epoch_data['samples_per_sec']:.2f} samples/s\n")
            
            for key, value in epoch_data.items():
                if key not in ['epoch', 'total_time', 'samples_per_sec']:
                    if isinstance(value, (list, tuple)):
                        f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: {value:.6f}\n")
            
            if phase == 'val':
                f.write(f"Best mAP@0.5: {self.best_map50:.4f} (Epoch {self.best_epoch})\n")
                f.write(f"Best mAP@0.5:0.95: {self.best_map:.4f}\n")
            
            f.write(f"{'='*60}\n") 
    
    def _update_csv_file(self, epoch_data: Dict[str, Any], phase: str):
        if not is_main_process():
            return
            
        try:
            df = pd.read_csv(self.results_file)
        except FileNotFoundError:
            df = pd.DataFrame()
        
        epoch = epoch_data['epoch']
        
        existing_row = df[df['epoch'] == epoch]
        
        if len(existing_row) > 0:
            row_idx = existing_row.index[0]
            
            if phase == 'train':
                df.loc[row_idx, 'train/loss'] = epoch_data.get('loss', 0)
                df.loc[row_idx, 'train/loss_ce'] = epoch_data.get('loss_vfl', 0)
                df.loc[row_idx, 'train/loss_bbox'] = epoch_data.get('loss_bbox', 0)
                df.loc[row_idx, 'train/loss_giou'] = epoch_data.get('loss_giou', 0)
                df.loc[row_idx, 'lr/pg0'] = epoch_data.get('lr', 0)
                df.loc[row_idx, 'time/train_epoch'] = epoch_data.get('total_time', 0)
            elif phase == 'val':
                if 'coco_eval_bbox' in epoch_data:
                    coco_stats = epoch_data['coco_eval_bbox']
                    if len(coco_stats) >= 12:
                        df.loc[row_idx, 'val/mAP_0.5'] = coco_stats[1]
                        df.loc[row_idx, 'val/mAP_0.5:0.95'] = coco_stats[0]
                        df.loc[row_idx, 'val/precision'] = coco_stats[1]
                        df.loc[row_idx, 'val/recall'] = coco_stats[7]
                        df.loc[row_idx, 'time/val_epoch'] = epoch_data.get('total_time', 0)
        else:
            new_row = {'epoch': epoch}
            
            if phase == 'train':
                new_row.update({
                    'train/loss': epoch_data.get('loss', 0),
                    'train/loss_ce': epoch_data.get('loss_vfl', 0),
                    'train/loss_bbox': epoch_data.get('loss_bbox', 0),
                    'train/loss_giou': epoch_data.get('loss_giou', 0),
                    'lr/pg0': epoch_data.get('lr', 0),
                    'time/train_epoch': epoch_data.get('total_time', 0)
                })
            elif phase == 'val':
                if 'coco_eval_bbox' in epoch_data:
                    coco_stats = epoch_data['coco_eval_bbox']
                    if len(coco_stats) >= 12:
                        new_row.update({
                            'val/mAP_0.5': coco_stats[1],
                            'val/mAP_0.5:0.95': coco_stats[0],
                            'val/recall': coco_stats[7],
                            'time/val_epoch': epoch_data.get('total_time', 0)
                        })
            
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        required_columns = [
            'epoch', 'train/loss', 'train/loss_ce', 'train/loss_bbox', 'train/loss_giou',
            'val/mAP_0.5', 'val/mAP_0.5:0.95', 'val/precision', 'val/recall',
            'lr/pg0', 'time/train_epoch', 'time/val_epoch'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df.sort_values('epoch').reset_index(drop=True)
        
        df.to_csv(self.results_file, index=False)
    
    def log_model_info(self, model: torch.nn.Module, device: torch.device):
        if not is_main_process():
            return {}
        
        model_info = {}
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info['total_params'] = total_params
        model_info['trainable_params'] = trainable_params
        model_info['model_size_mb'] = total_params * 4 / (1024 * 1024)
        model_info['device'] = str(device)
        model_info['num_classes'] = self.num_classes
        
        with open(self.model_info_file, 'w', encoding='utf-8') as f:
            f.write(f"Model Information\n")
            f.write(f"{'='*50}\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
            f.write(f"Device: {device}\n")
            f.write(f"Number of Classes: {self.num_classes}\n")
        
        return model_info
    
    def save_training_summary(self, model_info: Dict[str, Any], 
                            best_metrics: Dict[str, Any],
                            total_training_time: float):
        if not is_main_process():
            return
            
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RT-DETR Training Summary (YOLO Style)\n")
            f.write(f"{'='*60}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)\n")
            f.write(f"Best Epoch: {best_metrics.get('best_epoch', self.best_epoch)}\n")
            f.write(f"Best mAP@0.5: {best_metrics.get('best_map50', self.best_map50):.4f}\n")
            f.write(f"Best mAP@0.5:0.95: {best_metrics.get('best_map', self.best_map):.4f}\n")
            f.write(f"\nModel Information:\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
    
    def create_training_plots(self):
        if not is_main_process():
            return
            
        try:
            df = pd.read_csv(self.results_file)
            if df.empty:
                print("No training data available for plotting")
                return
            
            self._create_results_plot(df)
            self._create_loss_plot(df)
            self._create_map_plot(df)
            self._create_lr_plot(df)
            
            print(f"Training plots saved to {self.plots_dir}")
            
            self._cleanup_memory()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def _create_results_plot(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'RT-DETR Training Results - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(df['epoch'], df['train/loss'], 'b-', label='Total Loss', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['train/loss_ce'], 'r-', label='CE Loss', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['train/loss_bbox'], 'g-', label='BBox Loss', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['train/loss_giou'], 'm-', label='GIoU Loss', linewidth=2)
        axes[0, 0].set_title('Training Losses', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df['epoch'], df['val/mAP_0.5'], 'b-', label='mAP@0.5', linewidth=2)
        axes[0, 1].plot(df['epoch'], df['val/mAP_0.5:0.95'], 'r-', label='mAP@0.5:0.95', linewidth=2)
        axes[0, 1].set_title('Validation mAP', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(df['epoch'], df['val/precision'], 'b-', label='Precision', linewidth=2)
        axes[1, 0].plot(df['epoch'], df['val/recall'], 'r-', label='Recall', linewidth=2)
        axes[1, 0].set_title('Validation Precision/Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], 'b-', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'results.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_loss_plot(self, df: pd.DataFrame):
        fig = plt.figure(figsize=(12, 8))
        
        plt.plot(df['epoch'], df['train/loss'], 'b-', label='Total Loss', linewidth=2)
        plt.plot(df['epoch'], df['train/loss_ce'], 'r-', label='CE Loss', linewidth=2)
        plt.plot(df['epoch'], df['train/loss_bbox'], 'g-', label='BBox Loss', linewidth=2)
        plt.plot(df['epoch'], df['train/loss_giou'], 'm-', label='GIoU Loss', linewidth=2)
        
        plt.title('RT-DETR Training Losses', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'loss.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_map_plot(self, df: pd.DataFrame):
        fig = plt.figure(figsize=(12, 8))
        
        plt.plot(df['epoch'], df['val/mAP_0.5'], 'b-', label='mAP@0.5', linewidth=2)
        plt.plot(df['epoch'], df['val/mAP_0.5:0.95'], 'r-', label='mAP@0.5:0.95', linewidth=2)
        
        plt.title('RT-DETR Validation mAP', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mAP', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'map.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_lr_plot(self, df: pd.DataFrame):
        fig = plt.figure(figsize=(12, 8))
        
        plt.plot(df['epoch'], df['lr/pg0'], 'b-', linewidth=2)
        
        plt.title('RT-DETR Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def get_performance_stats(self):
        if not self.epoch_times:
            return {}
        
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)
        avg_step_time = np.mean(list(self.step_times)) if self.step_times else 0
        
        return {
            'total_training_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'avg_step_time': avg_step_time,
            'total_epochs': len(self.epochs),
            'best_map50': self.best_map50,
            'best_map': self.best_map,
            'best_epoch': self.best_epoch
        }


class YOLOEnhancedLogger:
    
    def __init__(self, save_dir: str, experiment_name: str = "rtdetr_experiment", 
                 num_classes: int = 80, log_interval: int = 50, max_step_history: int = 1000):
        self.processor = YOLOResultsProcessor(
            save_dir=save_dir,
            experiment_name=experiment_name,
            num_classes=num_classes,
            log_interval=log_interval,
            max_step_history=max_step_history
        )
    
    def log_train_step(self, epoch: int, step: int, metrics: Dict[str, float], 
                      lr: float, batch_size: int, time_per_step: float):
        self.processor.log_train_step(epoch, step, metrics, lr, batch_size, time_per_step)
    
    def log_train_epoch(self, epoch: int, metrics: Dict[str, float], 
                       total_time: float, samples_per_sec: float):
        self.processor.log_train_epoch(epoch, metrics, total_time, samples_per_sec)
    
    def log_val_epoch(self, epoch: int, metrics: Dict[str, float], 
                     total_time: float, samples_per_sec: float):
        self.processor.log_val_epoch(epoch, metrics, total_time, samples_per_sec)
    
    def log_model_info(self, model: torch.nn.Module, device: torch.device):
        return self.processor.log_model_info(model, device)
    
    def save_training_summary(self, model_info: Dict[str, Any], 
                            best_metrics: Dict[str, Any],
                            total_training_time: float):
        self.processor.save_training_summary(model_info, best_metrics, total_training_time)
    
    def create_training_plots(self):
        self.processor.create_training_plots()
    
    def get_performance_stats(self):
        return self.processor.get_performance_stats() 