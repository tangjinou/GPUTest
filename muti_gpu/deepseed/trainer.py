"""
训练器模块
包含模型创建、数据加载器创建等训练相关的辅助函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler
from model import SimpleNet
from dataset import DummyDataset


def create_model(rank, args):
    """
    创建并初始化模型（DeepSpeed会在initialize时包装）
    
    Args:
        rank: 当前进程的rank
        args: 训练参数
        
    Returns:
        未包装的模型（DeepSpeed会在initialize时处理）
    """
    try:
        import torch.distributed as dist
        
        # 检查CUDA设备是否可用
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        
        # 检查指定的GPU设备是否存在
        if rank >= torch.cuda.device_count():
            raise RuntimeError(f"GPU {rank} 不存在，只有 {torch.cuda.device_count()} 个GPU可用")
        
        # 确保CUDA设备已设置
        torch.cuda.set_device(rank)
        
        # 在创建模型之前，验证CUDA上下文是否正常工作
        try:
            test_tensor = torch.zeros(1, device=f'cuda:{rank}')
            _ = test_tensor + 1
            torch.cuda.synchronize(rank)
            del test_tensor
        except RuntimeError as cuda_error:
            error_msg = str(cuda_error)
            if 'driver version' in error_msg.lower() or 'cuda error' in error_msg.lower():
                raise RuntimeError(
                    f"[Rank {rank}] CUDA驱动版本不匹配，无法创建模型\n"
                    f"错误: {error_msg}\n"
                    f"请先解决CUDA版本兼容性问题"
                ) from cuda_error
            raise
        
        # 创建模型（移动到对应GPU）
        model = SimpleNet(
            in_features=args.features,
            hidden_size=args.hidden_size,
            num_classes=args.num_classes
        ).to(f'cuda:{rank}')
        
        # 确保所有进程的CUDA上下文都已完全初始化
        test_tensor = torch.zeros(10, 10, device=f'cuda:{rank}')
        _ = test_tensor @ test_tensor.T
        torch.cuda.synchronize(rank)
        del test_tensor
        torch.cuda.empty_cache()
        
        # 在所有进程间进行barrier
        dist.barrier()
        
        # 验证CUDA和NCCL协同工作
        test_tensor = torch.ones(1, device=f'cuda:{rank}')
        try:
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(rank)
            del test_tensor
        except Exception as e:
            error_msg = str(e)
            if 'driver version' in error_msg.lower() or 'cuda error' in error_msg.lower():
                raise RuntimeError(
                    f"[Rank {rank}] NCCL与CUDA协同工作失败: {error_msg}\n"
                    f"这通常表示CUDA驱动版本与PyTorch CUDA运行时版本不匹配\n"
                    f"PyTorch CUDA版本: {torch.version.cuda}\n"
                    f"请更新NVIDIA驱动或安装匹配的PyTorch版本"
                ) from e
            raise
        
        return model
    except Exception as e:
        print(f"[Rank {rank}] 创建模型时出错: {e}")
        raise


def create_dataloader(rank, world_size, args):
    """
    创建数据加载器
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        args: 训练参数
        
    Returns:
        DataLoader和DistributedSampler的元组
    """
    # 创建数据集
    dataset = DummyDataset(
        size=args.dataset_size, 
        features=args.features,
        noise_level=args.noise_level
    )
    
    # 使用DistributedSampler确保每个GPU处理不同的数据
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True  # 加速数据传输到GPU
    )
    
    return dataloader, sampler


def create_optimizer_and_scheduler(model, args):
    """
    创建优化器和学习率调度器
    注意：对于DeepSpeed，优化器会在initialize时创建
    
    Args:
        model: 模型
        args: 训练参数
        
    Returns:
        (optimizer, scheduler, criterion) 元组
    """
    criterion = nn.CrossEntropyLoss()
    
    # DeepSpeed会在initialize时创建优化器，这里先创建一个用于配置
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    
    return optimizer, scheduler, criterion


def train_one_epoch(model_engine, dataloader, criterion, rank, epoch, args):
    """
    训练一个epoch（DeepSpeed版本）
    
    Args:
        model_engine: DeepSpeed包装后的模型引擎
        dataloader: 数据加载器
        criterion: 损失函数
        rank: 当前进程的rank
        epoch: 当前epoch编号
        args: 训练参数
        
    Returns:
        平均损失
    """
    import torch.distributed as dist
    
    model_engine.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        # 将数据移动到对应GPU（使用non_blocking提高性能）
        batch_x = batch_x.to(f'cuda:{rank}', non_blocking=True)
        batch_y = batch_y.to(f'cuda:{rank}', non_blocking=True)
        
        # 前向传播
        output = model_engine(batch_x)
        loss = criterion(output, batch_y)
        
        # DeepSpeed反向传播和优化器更新
        model_engine.backward(loss)
        model_engine.step()
        
        # 使用detach()避免构建计算图，减少内存占用
        epoch_loss += loss.detach().item()
        num_batches += 1
        
        # 每N个batch打印一次（只在rank 0上打印）
        if batch_idx % args.log_interval == 0 and rank == 0:
            print(f'Epoch [{epoch}/{args.num_epochs}], '
                  f'Batch [{batch_idx}/{len(dataloader)}], '
                  f'Loss: {loss.detach().item():.4f}')
    
    # 计算平均损失（DeepSpeed已经处理了同步，这里只需要简单平均）
    # 注意：由于每个进程处理的数据量相同（DistributedSampler保证），
    # 直接平均即可，不需要allreduce（减少通信开销）
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss

