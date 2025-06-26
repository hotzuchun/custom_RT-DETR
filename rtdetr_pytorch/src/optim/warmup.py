
"""
by HO TZU CHUN
"""
from torch.optim.lr_scheduler import LRScheduler

from src.core import register


class Warmup(object):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int=-1, 
                 start_lr: float = None, end_lr: float = None) -> None:
        self.lr_scheduler = lr_scheduler
        self.warmup_end_values = [pg['lr'] for pg in lr_scheduler.optimizer.param_groups]
        self.last_step = last_step
        self.warmup_duration = warmup_duration
        
        self.start_lr = start_lr
        self.end_lr = end_lr
        
        if self.start_lr is not None:
            for pg in lr_scheduler.optimizer.param_groups:
                pg['lr'] = self.start_lr
        
        self.step()

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'lr_scheduler'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_warmup_factor(self, step, **kwargs):
        raise NotImplementedError

    def step(self, ):
        self.last_step += 1
        if self.last_step >= self.warmup_duration:
            return
            
        factor = self.get_warmup_factor(self.last_step)
        
        for i, pg in enumerate(self.lr_scheduler.optimizer.param_groups):
            if self.start_lr is not None and self.end_lr is not None:
                # 使用自定义的起始和终点学习率
                current_lr = self.start_lr + factor * (self.end_lr - self.start_lr)
                pg['lr'] = current_lr
            else:
                # 使用原来的方式：基于目标学习率的因子
                pg['lr'] = factor * self.warmup_end_values[i]
    
    def finished(self, ):
        if self.last_step >= self.warmup_duration:
            return True 
        return False


@register
class LinearWarmup(Warmup):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1,
                 start_lr: float = None, end_lr: float = None) -> None:
        super().__init__(lr_scheduler, warmup_duration, last_step, start_lr, end_lr)

    def get_warmup_factor(self, step):
        return min(1.0, (step + 1) / self.warmup_duration) 