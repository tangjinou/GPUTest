"""
分布式训练工具模块
包含分布式进程组的初始化和清理函数
"""

import os
import torch
import torch.distributed as dist


def setup(rank, world_size, master_addr='localhost', master_port='29500'):
    """
    初始化分布式进程组
    
    Args:
        rank: 当前进程的rank（GPU编号）
        world_size: 总进程数（GPU数量）
        master_addr: 主节点地址
        master_port: 主节点端口
    """
    # 在初始化分布式之前，先检查CUDA是否可用
    if not torch.cuda.is_available():
        raise RuntimeError(f"[Rank {rank}] CUDA不可用")
    
    # 检查指定的GPU设备是否存在
    if rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"[Rank {rank}] GPU {rank} 不存在，只有 {torch.cuda.device_count()} 个GPU可用"
        )
    
    # 设置CUDA设备并初始化CUDA上下文
    # 这必须在init_process_group之前完成，以确保CUDA驱动正确初始化
    try:
        # 设置当前设备
        torch.cuda.set_device(rank)
        
        # 检查CUDA驱动和运行时版本兼容性
        # 通过尝试获取设备属性来验证CUDA是否正常工作
        try:
            device_props = torch.cuda.get_device_properties(rank)
            # 尝试在GPU上执行一个操作来验证CUDA上下文
            test_tensor = torch.zeros(1, device=f'cuda:{rank}')
            # 执行多个操作来确保CUDA上下文完全初始化
            _ = test_tensor + 1
            _ = test_tensor * 2
            # 同步以确保操作完成
            torch.cuda.synchronize(rank)
            del test_tensor
            torch.cuda.empty_cache()
        except RuntimeError as cuda_error:
            error_msg = str(cuda_error)
            if 'driver version' in error_msg.lower() or 'cuda error' in error_msg.lower():
                # 获取版本信息用于诊断
                try:
                    import subprocess
                    nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                                         stderr=subprocess.DEVNULL).decode().strip()
                except:
                    nvidia_smi = "无法获取"
                
                raise RuntimeError(
                    f"[Rank {rank}] CUDA驱动版本不匹配！\n"
                    f"错误详情: {error_msg}\n"
                    f"NVIDIA驱动版本: {nvidia_smi}\n"
                    f"PyTorch CUDA版本: {torch.version.cuda}\n"
                    f"解决方案:\n"
                    f"1. 检查驱动版本: nvidia-smi\n"
                    f"2. 更新NVIDIA驱动到最新版本\n"
                    f"3. 或安装与驱动版本匹配的PyTorch版本"
                ) from cuda_error
            raise
    except Exception as e:
        raise RuntimeError(
            f"[Rank {rank}] GPU {rank} 初始化失败: {e}\n"
            f"提示: 请检查CUDA驱动版本是否与PyTorch的CUDA运行时版本匹配\n"
            f"PyTorch CUDA版本: {torch.version.cuda}"
        ) from e
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # 设置NCCL环境变量以帮助解决CUDA版本问题
    # 这些变量可以帮助NCCL更好地处理CUDA初始化
    os.environ.setdefault('NCCL_IB_DISABLE', '1')  # 禁用InfiniBand，使用以太网
    os.environ.setdefault('NCCL_P2P_DISABLE', '0')  # 启用P2P通信
    os.environ.setdefault('NCCL_SHM_DISABLE', '0')  # 启用共享内存
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')  # 使用本地回环接口
    # 使用新的环境变量名称（避免弃用警告）
    os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    # 移除旧的弃用变量（如果存在）
    os.environ.pop('NCCL_BLOCKING_WAIT', None)
    os.environ.pop('NCCL_ASYNC_ERROR_HANDLING', None)
    # 添加NCCL调试和兼容性选项
    # 如果CUDA驱动版本不匹配，尝试使用更宽松的检查
    os.environ.setdefault('NCCL_DEBUG', 'WARN')  # 设置为WARN以减少输出，INFO用于详细调试
    # 尝试禁用一些可能导致CUDA版本检查失败的特性
    os.environ.setdefault('NCCL_P2P_LEVEL', 'NVL')  # 使用NVLink进行P2P通信
    
    # 在初始化进程组之前，再次确保CUDA上下文完全就绪
    # 执行一个更复杂的CUDA操作来确保驱动完全加载
    try:
        # 创建一个更大的tensor并进行计算，确保CUDA驱动完全初始化
        large_tensor = torch.randn(100, 100, device=f'cuda:{rank}')
        _ = large_tensor @ large_tensor.T
        torch.cuda.synchronize(rank)
        del large_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(
            f"[Rank {rank}] CUDA上下文初始化失败: {e}\n"
            f"这通常表示CUDA驱动版本与PyTorch CUDA运行时版本不匹配"
        ) from e
    
    # 检查NCCL可用性
    if not dist.is_nccl_available():
        raise RuntimeError(f"[Rank {rank}] NCCL不可用，无法进行多GPU分布式训练")
    
    # 初始化进程组
    # 注意：在spawn模式下，每个进程都需要独立初始化CUDA
    try:
        dist.init_process_group(
            backend='nccl',  # 使用NCCL后端（适用于多GPU）
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.default_pg_timeout  # 使用默认超时
        )
    except Exception as e:
        error_msg = str(e)
        # 检查是否是CUDA驱动版本问题
        if 'CUDA driver version' in error_msg or 'cuda error' in error_msg.lower() or 'driver version is insufficient' in error_msg.lower():
            # 获取详细的版本信息用于诊断
            try:
                import subprocess
                driver_info = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader'],
                    stderr=subprocess.DEVNULL
                ).decode().strip().split('\n')[rank] if rank < 4 else "无法获取"
            except:
                driver_info = "无法获取"
            
            raise RuntimeError(
                f"[Rank {rank}] CUDA/NCCL初始化失败 - 驱动版本不匹配\n"
                f"错误详情: {error_msg}\n"
                f"\n诊断信息:\n"
                f"  - PyTorch版本: {torch.__version__}\n"
                f"  - PyTorch CUDA版本: {torch.version.cuda}\n"
                f"  - GPU {rank} 信息: {driver_info}\n"
                f"  - NCCL版本: {torch.cuda.nccl.version() if torch.cuda.is_available() else 'N/A'}\n"
                f"\n可能的解决方案:\n"
                f"1. 更新NVIDIA驱动到最新版本（推荐）\n"
                f"   - 检查驱动版本: nvidia-smi\n"
                f"   - 驱动575.57.08可能不完全支持CUDA 12.8\n"
                f"   - 建议更新到支持CUDA 12.8的驱动版本\n"
                f"2. 安装与当前驱动版本匹配的PyTorch版本\n"
                f"   - 当前驱动可能支持CUDA 12.1或12.4\n"
                f"   - 可以尝试安装: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
                f"3. 检查CUDA兼容性:\n"
                f"   - 运行: python -c 'import torch; print(torch.__version__, torch.version.cuda)'\n"
                f"   - 运行: nvidia-smi\n"
            ) from e
        # 检查是否是NCCL特定问题
        elif 'NCCL' in error_msg or 'nccl' in error_msg.lower():
            raise RuntimeError(
                f"[Rank {rank}] NCCL初始化失败: {error_msg}\n"
                f"这可能是NCCL配置问题或CUDA驱动版本不匹配导致的\n"
                f"建议:\n"
                f"1. 检查CUDA驱动版本是否与PyTorch CUDA版本匹配\n"
                f"2. 尝试设置环境变量: export NCCL_DEBUG=INFO\n"
                f"3. 检查NCCL版本: python -c 'import torch; print(torch.cuda.nccl.version())'"
            ) from e
        raise


def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()

