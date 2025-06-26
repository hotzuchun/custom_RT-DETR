
'''

by lyuwenyu
'''
import os
import sys
import time
import math
import random
import datetime
import gc
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import torch.distributed as torch_dist
from src.misc import dist
from torch.utils.data import DataLoader, DistributedSampler

from src.core.config import BaseConfig
from src.misc.custom_processor import YOLOEnhancedLogger
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        # 初始化YOLO风格的日志器
        self.enhanced_logger = YOLOEnhancedLogger(
            save_dir=self.output_dir,
            experiment_name=f"rtdetr_experiment_{args.epoches}epochs",
            num_classes=80,  # 默认COCO类别数
            log_interval=50
        )
        
        # 记录模型信息
        model_info = self.enhanced_logger.log_model_info(self.model, self.device)
        
        # 初始化预热调度器
        self.lr_warmup_scheduler = getattr(self.cfg, 'lr_warmup_scheduler', None)
        
        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler,
                enhanced_logger=self.enhanced_logger, lr_warmup_scheduler=self.lr_warmup_scheduler)

            # 只有在预热完成后才步进主学习率调度器
            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir,
                enhanced_logger=self.enhanced_logger, epoch=epoch
            )

            # 更新最佳统计信息
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            # 模型保存逻辑 - 移到验证之后
            if self.output_dir:
                # 保存最新检查点
                dist.save_on_master(self.state_dict(epoch), self.output_dir / 'latest.pth')
                
                # 保存最佳检查点
                if best_stat['epoch'] == epoch:
                    dist.save_on_master(self.state_dict(epoch), self.output_dir / 'best.pth')
                    print(f"保存最佳模型检查点，epoch {epoch}")
                
                # 定期保存检查点（每50个epoch）
                if epoch % 50 == 0:
                    dist.save_on_master(self.state_dict(epoch), self.output_dir / f'{epoch:03}.pth')
                    print(f"保存定期检查点，epoch {epoch}")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            # 每个epoch结束后自动清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理未使用的显存
            gc.collect()  # 强制进行python垃圾回收
            print(f"Epoch {epoch} 完成，已清理内存")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        # 保存训练总结和创建可视化图表
        if dist.is_main_process():
            self.enhanced_logger.save_training_summary(
                model_info=model_info,
                best_metrics=best_stat,
                total_training_time=total_time
            )
            self.enhanced_logger.create_training_plots()
            print(f"训练完成！详细日志和可视化图表已保存到: {self.output_dir}")


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
