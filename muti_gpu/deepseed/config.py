"""
配置模块
包含参数解析和验证逻辑
"""

import argparse
import torch


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='多GPU DeepSpeed训练')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='每个GPU的batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--lr_step', type=int, default=5, help='学习率衰减步数')
    
    # 模型参数
    parser.add_argument('--features', type=int, default=10, help='输入特征数')
    parser.add_argument('--hidden_size', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数')
    
    # 数据参数
    parser.add_argument('--dataset_size', type=int, default=10000, help='数据集大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--noise_level', type=float, default=0.1, 
                       help='数据噪声水平（0-1，越小越容易学习）')
    
    # 分布式参数
    parser.add_argument('--world_size', type=int, default=4, help='GPU数量')
    parser.add_argument('--master_addr', type=str, default='localhost', help='主节点地址')
    parser.add_argument('--master_port', type=str, default='29500', help='主节点端口')
    
    # DeepSpeed参数
    parser.add_argument('--deepspeed_config', type=str, default=None, 
                       help='DeepSpeed配置文件路径（默认使用deepspeed_config.json）')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='梯度累积步数')
    
    # 日志参数
    parser.add_argument('--log_interval', type=int, default=10, help='日志打印间隔')
    
    return parser.parse_args()


def validate_args(args):
    """
    验证参数的有效性
    
    Args:
        args: 训练参数
        
    Returns:
        bool: 如果参数有效返回True，否则返回False
    """
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA，无法使用GPU训练")
        return False
    
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus < args.world_size:
        print(f"警告: 请求使用 {args.world_size} 个GPU，但只有 {num_gpus} 个可用")
        args.world_size = num_gpus
    
    if args.world_size <= 0:
        print("错误: GPU数量必须大于0")
        return False
    
    # 检查每个GPU是否可用（测试CUDA版本兼容性）
    print("正在检查GPU可用性...")
    failed_gpus = []
    for gpu_id in range(args.world_size):
        try:
            # 尝试在GPU上创建tensor
            torch.cuda.set_device(gpu_id)
            test_tensor = torch.zeros(1, device=f'cuda:{gpu_id}')
            # 尝试执行一个简单的CUDA操作
            result = test_tensor + 1
            del test_tensor, result
            torch.cuda.empty_cache()
            print(f"  GPU {gpu_id}: ✓ 可用")
        except Exception as e:
            error_msg = str(e)
            print(f"  GPU {gpu_id}: ✗ 不可用 - {error_msg}")
            failed_gpus.append((gpu_id, error_msg))
    
    if failed_gpus:
        print("\n错误: 部分GPU不可用或CUDA版本不匹配")
        print("详细信息:")
        for gpu_id, error in failed_gpus:
            print(f"  GPU {gpu_id}: {error}")
        print("\n诊断信息:")
        print(f"  PyTorch CUDA版本: {torch.version.cuda}")
        print(f"  PyTorch版本: {torch.__version__}")
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("\n  NVIDIA驱动信息:")
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines[:5]):
                    if line.strip():
                        print(f"    {line}")
        except Exception as e:
            print(f"  无法获取NVIDIA驱动信息: {e}")
        
        print("\n解决方案:")
        print("  1. 检查CUDA驱动版本: nvidia-smi")
        print("  2. 检查PyTorch CUDA版本: python -c 'import torch; print(torch.version.cuda)'")
        print("  3. 确保CUDA驱动版本 >= PyTorch CUDA版本")
        print("  4. 如果版本不匹配，请更新NVIDIA驱动或重新安装匹配的PyTorch版本")
        return False
    
    print("所有GPU检查通过 ✓\n")
    return True


def print_training_info(args):
    """
    打印训练信息
    
    Args:
        args: 训练参数
    """
    print(f"开始 {args.world_size} 卡DeepSpeed训练")
    print(f"使用GPU: {list(range(args.world_size))}")
    print(f"训练参数: epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print('=' * 60)

