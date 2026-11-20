# MaIR / Spatial-Mamba 移植对照

FP32，CUDA，TF32 关闭；各 stage 使用 256×256、128×128、64×64 对应形状及两种 shift，复制相同权重。相对误差定义为逐元素 `abs(port-reference)/(abs(reference)+1e-6)` 后取最大值；梯度使用相同随机输出梯度。下表为所有 stage 和 shift 的最大值。

| 变体 | 前向最大绝对差 | 前向最大相对差 | 输入梯度最大绝对差 | 输入梯度最大相对差 | 权重 strict | hook 模块/调用 |
|---|---:|---:|---:|---:|---|---|
| MaIR 核心 | 0 | 0 | 2.384e-7 | 0.03942 | 通过 | 官方 VMM 22/22 |
| MaIR 整块 | 0 | 0 | 4.768e-7 | 0.04390 | 通过 | 官方 VMM 22/22；官方 VSSBlock 0 |
| Spatial-Mamba 核心 | 0 | 0 | 2.384e-7 | 0.01800 | 通过 | 官方 StructureAwareSSM 22/22 |
| Spatial-Mamba 整块 | 0 | 0 | 9.537e-7 | 0.004094 | 通过 | 官方 SpatialMambaBlock 22/22 |

MaIR 整块由 `_MaIRVSSBlock` 重建外层逻辑，其官方 VSSBlock 的实例数为 0；因此不能称 22 处直接运行官方完整块。对照使用独立载入的官方 VSSBlock、原始扫描索引及同一权重。

Spatial-Mamba 官方 depthwise 扩展只导出前向，不导出 Python 反向所需函数；输入梯度对照使用 PyTorch 的精确卷积伴随，不能称为原版扩展反向验证。扩展副本仅增加 PyTorch 所需的头文件，按 sm_120 编译。

官方模型复现其论文数字：未做；未确认本地同时具备官方预训练权重及对应测试集。详细形状和逐项误差见 `A_PORT_CHECK.json`。
