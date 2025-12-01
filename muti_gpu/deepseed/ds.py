"""
多GPU DeepSpeed 训练主程序
支持多GPU分布式训练，使用DeepSpeed进行模型参数、梯度和优化器状态分片
"""

import torch
import torch.multiprocessing as mp
import deepspeed
import os
from distributed import setup, cleanup
from trainer import create_model, create_dataloader, create_optimizer_and_scheduler, train_one_epoch
from config import parse_args, validate_args, print_training_info


def train(rank, world_size, args):
    """
    训练函数，在每个GPU上运行
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        args: 训练参数
    """
    try:
        # 1. 设置分布式环境
        setup(rank, world_size, args.master_addr, args.master_port)
        
        # 1.5. 验证CUDA和NCCL协同工作
        # 在spawn模式下，每个子进程都需要独立验证CUDA上下文
        import torch.distributed as dist
        try:
            # 确保CUDA设备已设置
            torch.cuda.set_device(rank)
            
            # 创建一个tensor并执行allreduce来验证NCCL和CUDA协同工作
            test_tensor = torch.ones(1, device=f'cuda:{rank}') * (rank + 1)
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(rank)
            
            # 验证结果（所有进程的tensor值应该相同）
            expected_sum = sum(range(1, world_size + 1))
            if rank == 0 and abs(test_tensor.item() - expected_sum) > 1e-5:
                print(f"[Rank {rank}] 警告: NCCL allreduce结果异常: {test_tensor.item()} != {expected_sum}")
            
            del test_tensor
        except Exception as e:
            error_msg = str(e)
            if 'driver version' in error_msg.lower() or 'cuda error' in error_msg.lower():
                raise RuntimeError(
                    f"[Rank {rank}] CUDA/NCCL验证失败: {error_msg}\n"
                    f"这通常表示CUDA驱动版本与PyTorch CUDA运行时版本不匹配\n"
                    f"请更新NVIDIA驱动或安装匹配的PyTorch版本"
                ) from e
            raise
        
        # 2. 设置随机种子以确保可复现性
        torch.manual_seed(42 + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + rank)
        
        # 3. 创建模型
        model = create_model(rank, args)
        
        # 4. 创建数据加载器
        dataloader, sampler = create_dataloader(rank, world_size, args)
        
        # 5. 创建优化器和调度器（DeepSpeed会在initialize时重新创建）
        optimizer, scheduler, criterion = create_optimizer_and_scheduler(model, args)
        
        # 6. 设置DeepSpeed所需的环境变量
        # DeepSpeed需要LOCAL_RANK环境变量，在单机多卡情况下等于rank
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # 7. 创建或加载DeepSpeed配置
        deepspeed_config = get_deepspeed_config(args)
        
        # 8. 初始化DeepSpeed引擎
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=deepspeed_config,
            args=args
        )
        
        # 9. 训练循环
        for epoch in range(args.num_epochs):
            # 设置epoch，确保每个epoch的数据顺序不同
            sampler.set_epoch(epoch)
            
            # 训练一个epoch
            avg_loss = train_one_epoch(
                model_engine, dataloader, criterion, 
                rank, epoch, args
            )
            
            # 更新学习率（DeepSpeed引擎会自动处理）
            scheduler.step()
            
            # 只在rank 0上打印epoch信息
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch}/{args.num_epochs}] 完成 - '
                      f'平均损失: {avg_loss:.4f}, '
                      f'学习率: {current_lr:.6f}')
                print('-' * 60)
        
        # 10. 训练完成
        if rank == 0:
            print('训练完成！')
    
    except Exception as e:
        if rank == 0:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        # 确保总是清理分布式进程组
        try:
            cleanup()
        except:
            pass


def check_cuda_compatibility():
    """
    在启动训练前检查CUDA兼容性
    提前发现CUDA驱动版本不匹配的问题
    """
    if not torch.cuda.is_available():
        print("警告: CUDA不可用")
        return False
    
    try:
        # 尝试在GPU 0上执行操作来验证CUDA是否正常工作
        test_tensor = torch.zeros(1, device='cuda:0')
        _ = test_tensor + 1
        torch.cuda.synchronize(0)
        del test_tensor
        torch.cuda.empty_cache()
        
        # 获取版本信息
        print(f"CUDA检查通过:")
        print(f"  PyTorch CUDA版本: {torch.version.cuda}")
        print(f"  CUDA设备数量: {torch.cuda.device_count()}")
        try:
            import subprocess
            nvidia_smi = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL
            ).decode().strip().split('\n')[0]
            print(f"  NVIDIA驱动: {nvidia_smi}")
        except:
            print(f"  NVIDIA驱动: 无法获取（请运行 nvidia-smi 查看）")
        
        return True
    except RuntimeError as e:
        error_msg = str(e)
        if 'driver version' in error_msg.lower() or 'cuda error' in error_msg.lower():
            print("=" * 60)
            print("错误: CUDA驱动版本不匹配！")
            print("=" * 60)
            print(f"错误详情: {error_msg}")
            print(f"PyTorch CUDA版本: {torch.version.cuda}")
            try:
                import subprocess
                nvidia_smi = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                print(f"NVIDIA驱动版本: {nvidia_smi}")
            except:
                print("NVIDIA驱动版本: 无法获取（请运行 nvidia-smi 查看）")
            print("\n解决方案:")
            print("1. 更新NVIDIA驱动到最新版本:")
            print("   sudo apt update && sudo apt install nvidia-driver-latest")
            print("2. 或安装与驱动版本匹配的PyTorch版本")
            print("3. 检查PyTorch安装: python -c 'import torch; print(torch.__version__, torch.version.cuda)'")
            print("=" * 60)
        else:
            print(f"CUDA初始化失败: {error_msg}")
        return False
    except Exception as e:
        print(f"CUDA检查失败: {e}")
        return False


def get_deepspeed_config(args):
    """
    创建或加载DeepSpeed配置
    
    Args:
        args: 训练参数
        
    Returns:
        DeepSpeed配置字典或文件路径
    """
    # 如果指定了配置文件，直接返回路径
    if args.deepspeed_config and os.path.exists(args.deepspeed_config):
        return args.deepspeed_config
    
    # 否则创建默认配置
    deepspeed_config = {
        "train_batch_size": args.batch_size * args.world_size,
        "train_micro_batch_size_per_gpu": args.batch_size,  # 每个GPU的micro batch size
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "StepLR",
            "params": {
                "step_size": args.lr_step,
                "gamma": 0.1
            }
        },
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": False
        },
        "zero_optimization": {
            # 注意：对于小模型（如hidden_size < 1000），ZeRO-2可能比ZeRO-3更快
            # 因为ZeRO-3的通信开销（参数allgather）可能超过其内存节省带来的收益
            # 如果模型较大（>1B参数），ZeRO-3会更合适
            "stage": 3,  # 使用ZeRO-3，分片参数、优化器状态和梯度
            "offload_optimizer": {
                "device": "none",  # 可选: "cpu" 或 "none"，"cpu"可以进一步节省GPU内存
                "pin_memory": True
            },
            "offload_param": {
                "device": "none",  # 可选: "cpu" 或 "none"，"cpu"可以进一步节省GPU内存
                "pin_memory": True
            },
            "overlap_comm": True,  # 重叠通信和计算
            "contiguous_gradients": True,
            "round_robin_gradients": True,  # 轮询梯度，提高通信效率
            # ZeRO-3 优化参数（根据模型大小调整）
            "sub_group_size": 1e8,  # 减小sub_group_size以提高通信效率
            "reduce_bucket_size": 5e7,  # 明确设置reduce bucket大小
            "stage3_prefetch_bucket_size": 5e7,  # 参数预取桶大小
            "stage3_param_persistence_threshold": 1e5,  # 参数持久化阈值（较小的值减少内存占用）
            "stage3_max_live_parameters": 1e8,  # 最大活跃参数数（减小以减少内存碎片）
            "stage3_max_reuse_distance": 1e8,  # 最大重用距离
            "gather_16bit_weights_on_model_save": False  # 保存时不收集16位权重，减少开销（使用新参数名）
        },
        "wall_clock_breakdown": False,
        # 性能优化选项
        "steps_per_print": 1000  # 减少日志输出频率
    }
    
    return deepspeed_config


def main():
    """主函数"""
    try:
        # 1. 解析参数
        args = parse_args()
        
        # 2. 验证参数
        if not validate_args(args):
            return
        
        # 3. 在启动前检查CUDA兼容性
        print("正在检查CUDA兼容性...")
        if not check_cuda_compatibility():
            print("\nCUDA兼容性检查失败，训练无法继续。")
            print("请先解决CUDA版本兼容性问题。")
            return
        
        # 4. 打印训练信息
        print_training_info(args)
        
        # 5. 设置环境变量以确保CUDA和NCCL正确初始化
        import os
        # 设置NCCL环境变量（这些会在子进程中继承）
        os.environ.setdefault('NCCL_IB_DISABLE', '1')
        # 移除NCCL_SOCKET_IFNAME限制，让NCCL自动选择最佳网络接口
        # os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
        # 使用异步非阻塞模式提高性能
        os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '0')  # 非阻塞模式
        os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
        # 启用NCCL性能优化
        os.environ.setdefault('NCCL_TREE_THRESHOLD', '0')  # 对小规模通信使用ring算法
        os.environ.setdefault('NCCL_LAUNCH_MODE', 'PARALLEL')  # 并行启动模式
        # 如果需要调试，可以启用以下选项
        # os.environ.setdefault('NCCL_DEBUG', 'INFO')
        # os.environ.setdefault('NCCL_DEBUG_SUBSYS', 'ALL')
        
        # 检查NCCL可用性
        if not torch.distributed.is_nccl_available():
            print("警告: NCCL不可用，将无法进行多GPU训练")
            return
        else:
            try:
                nccl_version = torch.cuda.nccl.version()
                print(f"NCCL版本: {nccl_version[0]}.{nccl_version[1]}.{nccl_version[2]}")
            except:
                print("无法获取NCCL版本信息")
        
        # 6. 设置multiprocessing启动方法
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # 7. 启动多进程训练
        mp.spawn(
            train,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

