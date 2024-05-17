import torch

class EarlyStopping:

    def __init__(self, patience=5, min_delta=0):

        self.patience = patience  # 容忍的epoch数量，即在验证集上性能没有改善的情况下最多等待的epoch数量
        self.min_delta = min_delta  # 最小性能改善的阈值
        self.counter = 0  # 计数器，用于计算连续没有改善的epoch数量
        self.best_score = None  # 最佳性能得分
        self.early_stop = False  # 是否停止训练的标志

    def __call__(self, val_loss):

        if self.best_score is None:  # 第一个epoch的情况
            self.best_score = val_loss

        elif val_loss < self.best_score - self.min_delta:  # 有足够的改善
            self.best_score = val_loss
            self.counter = 0  # 重置计数器
        else:  # 没有足够的改善
            self.counter += 1
            if self.counter >= self.patience:  # 达到容忍的上限
                self.early_stop = True


class ModelCheckpoint:
    def __init__(self, path, filename):
        self.path = path
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            print(f'Folder [{self.path}] created successfully.')
        else:
            print(f'Folder [{self.path}] already exists.')
        self.filename = filename
        self.best_loss = float('inf')  # 初始化最佳损失为正无穷

    def save(self, model, optimizer, epoch, loss, args):
        if loss < self.best_loss:  # 如果当前损失比最佳损失小
            self.best_loss = loss  # 更新最佳损失
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'args': args
            }
            file_path = self.path / self.filename
            torch.save(checkpoint, file_path)  # 保存checkpoint
            print(f'Save checkpoint {file_path} with loss [{loss:.4f}]')

    def load(self, model, device, optimizer=None):
        file_path = self.path / self.filename
        checkpoint = torch.load(file_path, map_location=device)  # 加载checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        print(f'Load checkpoint from {file_path}')
        return checkpoint['epoch']  # 返回epoch和损失

    def load_args_from_ckpt(self, args):
        file_path = self.path / self.filename
        checkpoint = torch.load(file_path, map_location='cpu')
        for key, value in vars(checkpoint['args']).items():
            if key not in args.__dict__.keys():
                setattr(args, key, value)
        return args

