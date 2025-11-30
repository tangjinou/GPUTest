# GPUTest

GPU测试和性能评估项目，包含多GPU训练和NCCL测试工具。

## 项目结构

- `muti_gpu/` - 多GPU训练实现
  - `ddp/` - Distributed Data Parallel (DDP) 实现
  - `fsdp/` - Fully Sharded Data Parallel (FSDP) 实现
- `nccl-tests/` - NCCL性能测试工具
- `test.ipynb` - 测试和实验笔记本

## 功能特性

### 多GPU训练
- **DDP**: 使用PyTorch的DistributedDataParallel进行多GPU训练
- **FSDP**: 使用Fully Sharded Data Parallel进行大规模模型训练

### NCCL测试
- 包含NCCL官方测试工具，用于验证和评估NCCL通信性能
- 支持多种集合通信操作：all_reduce, all_gather, broadcast等

## 使用说明

### 多GPU训练

#### DDP训练
```bash
cd muti_gpu/ddp
python ddp.py
```

#### FSDP训练
```bash
cd muti_gpu/fsdp
python fsdp.py
```

### NCCL测试

参考 `nccl-tests/README.md` 了解详细使用方法。

## 依赖

- PyTorch
- CUDA
- NCCL
- MPI (可选，用于多节点测试)

## 许可证

请参考各子目录的许可证文件。

