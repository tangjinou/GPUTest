"""
训练器模块
包含模型创建、数据加载器创建等训练相关的辅助函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.utils.data import DataLoader, DistributedSampler
from model import SimpleNet
from dataset import DummyDataset


def create_model(rank, args):
    """
    创建并初始化模型，使用FSDP包装
    
    Args:
        rank: 当前进程的rank
        args: 训练参数
        
    Returns:
        FSDP包装后的模型
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
        
        # 创建模型（不移动到GPU，FSDP会处理）
        model = SimpleNet(
            in_features=args.features,
            hidden_size=args.hidden_size,
            num_classes=args.num_classes
        )
        
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
        
        # 配置FSDP策略
        # 使用FULL_SHARD策略，将参数、梯度和优化器状态都分片
        sharding_strategy = ShardingStrategy.FULL_SHARD
        
        # 配置混合精度（可选，如果需要可以启用）
        # mixed_precision = MixedPrecision(
        #     param_dtype=torch.float16,
        #     reduce_dtype=torch.float16,
        #     buffer_dtype=torch.float16,
        # )
        mixed_precision = None  # 使用全精度训练
        
        # 使用FSDP包装模型
        try:
            fsdp_model = FSDP(
                model,
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                device_id=rank,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                forward_prefetch=True,
                limit_all_gathers=True,  # 优化性能
            )
        except RuntimeError as fsdp_error:
            error_msg = str(fsdp_error)
            if 'driver version' in error_msg.lower() or 'cuda error' in error_msg.lower() or 'NCCL' in error_msg:
                try:
                    import subprocess
                    nvidia_smi = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=driver_version,compute_cap', '--format=csv,noheader'],
                        stderr=subprocess.DEVNULL
                    ).decode().strip().split('\n')[rank] if rank < 4 else "无法获取"
                except:
                    nvidia_smi = "无法获取"
                
                raise RuntimeError(
                    f"[Rank {rank}] FSDP初始化失败 - CUDA/NCCL错误\n"
                    f"错误详情: {error_msg}\n"
                    f"诊断信息:\n"
                    f"  - PyTorch CUDA版本: {torch.version.cuda}\n"
                    f"  - PyTorch版本: {torch.__version__}\n"
                    f"  - GPU {rank} 信息: {nvidia_smi}\n"
                    f"  - CUDA设备数量: {torch.cuda.device_count()}\n"
                    f"\n可能的解决方案:\n"
                    f"1. 更新NVIDIA驱动到最新版本（推荐）\n"
                    f"2. 安装与当前驱动版本匹配的PyTorch版本\n"
                    f"3. 检查CUDA驱动版本: nvidia-smi\n"
                    f"4. 检查PyTorch CUDA版本: python -c 'import torch; print(torch.version.cuda)'\n"
                    f"5. 确保PyTorch版本 >= 2.0.0（FSDP需要）\n"
                    f"6. 如果问题持续，尝试重新安装PyTorch: pip install torch --force-reinstall"
                ) from fsdp_error
            raise
        
        return fsdp_model
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
    注意：对于FSDP，优化器需要在FSDP包装之后创建
    
    Args:
        model: FSDP包装后的模型
        args: 训练参数
        
    Returns:
        (optimizer, scheduler, criterion) 元组
    """
    criterion = nn.CrossEntropyLoss()
    
    # FSDP模型需要访问sharded参数，使用model.parameters()即可
    # FSDP会自动处理参数分片
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    
    return optimizer, scheduler, criterion


def train_one_epoch(model, dataloader, optimizer, criterion, rank, epoch, args):
    """
    训练一个epoch（FSDP版本）
    
    Args:
        model: FSDP包装后的模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        rank: 当前进程的rank
        epoch: 当前epoch编号
        args: 训练参数
        
    Returns:
        平均损失
    """
    import torch.distributed as dist
    
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        # 将数据移动到对应GPU
        batch_x = batch_x.to(rank, non_blocking=True)
        batch_y = batch_y.to(rank, non_blocking=True)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # 每N个batch打印一次（只在rank 0上打印）
        if batch_idx % args.log_interval == 0 and rank == 0:
            print(f'Epoch [{epoch}/{args.num_epochs}], '
                  f'Batch [{batch_idx}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}')
    
    # 计算平均损失（FSDP会自动同步，但为了确保一致性，可以手动allreduce）
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    
    # 在所有进程间同步平均损失
    loss_tensor = torch.tensor(avg_loss, device=f'cuda:{rank}')
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor.item() / dist.get_world_size()
    
    return avg_loss

