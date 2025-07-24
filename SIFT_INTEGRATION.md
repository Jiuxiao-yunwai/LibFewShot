# SIFT Model Integration Guide

## 概述

SIFT (Semantic-aware Interactive Feature Transfer) 模型已成功集成到LibFewShot框架中。SIFT是一个用于少样本学习的语义感知交互特征转移方法。

## 安装依赖

首先安装SIFT所需的额外依赖：

```bash
pip install pulp>=2.0
```

或者更新整个环境：

```bash
pip install -r requirements.txt
```

## 配置文件

SIFT模型支持多种配置选项：

### 基本配置 (config/sift.yaml)
```yaml
includes:
  - headers/data.yaml
  - headers/device.yaml
  - headers/misc.yaml
  - headers/optimizer.yaml
  - backbones/Conv32F.yaml
  - classifiers/SIFT.yaml
```

### 分类器配置 (config/classifiers/SIFT.yaml)
```yaml
classifier:
  name: SIFT
  kwargs:
    mode: dc  # dc, st, ns
    classifier_method: metric  # metric, gradient, nonparam
    setting: in  # in (inductive), tran (transductive)
    lr: 0.001
    grad_lr: 0.01
    ablation: no  # no, enc_recon, dec_recon, cpt, all
    cls: lr  # lr, svm, knn (for nonparam classifier_method)
```

## 配置参数说明

### 模式 (mode)
- `dc`: Domain Conversion - 域转换模式
- `st`: Semantic Transfer - 语义转移模式 
- `ns`: Novel Support - 新颖支持模式

### 分类器方法 (classifier_method)
- `metric`: 基于度量的分类器（推荐）
- `gradient`: 基于梯度的分类器
- `nonparam`: 非参数分类器（LR/SVM/KNN）

### 设置 (setting)
- `in`: Inductive setting - 归纳设置
- `tran`: Transductive setting - 转导设置

### 消融研究 (ablation)
- `no`: 使用所有组件
- `enc_recon`: 移除编码器重构损失
- `dec_recon`: 移除解码器重构损失
- `cpt`: 移除紧凑性损失
- `all`: 仅使用基础损失

## 使用方法

### 1. 基本训练
```python
from core.config import Config
from core import Trainer

# 加载配置
config = Config("./config/sift.yaml").get_config_dict()

# 创建训练器
trainer = Trainer(0, config)
trainer.train_loop(0)
```

### 2. 自定义配置运行
```python
# 修改run_trainer.py中的配置
config = Config("./config/sift.yaml").get_config_dict()

# 或者在代码中直接修改参数
config['classifier']['kwargs']['mode'] = 'st'
config['classifier']['kwargs']['setting'] = 'tran'
```

### 3. 测试模型集成
```bash
python test_sift.py
```

## 不同模式的详细说明

### DC模式 (Domain Conversion)
最简单的模式，适合快速测试：
```yaml
classifier:
  name: SIFT
  kwargs:
    mode: dc
    classifier_method: metric
    setting: in
```

### ST模式 (Semantic Transfer)
完整的SIFT模型，包含语义转移：
```yaml
classifier:
  name: SIFT
  kwargs:
    mode: st
    classifier_method: metric
    setting: tran
    lr: 0.001
```

### NS模式 (Novel Support)
对比学习版本：
```yaml
classifier:
  name: SIFT
  kwargs:
    mode: ns
    classifier_method: metric
    setting: tran
```

## 性能调优建议

1. **学习率调整**：
   - `lr`: 主要学习率，建议范围 [0.0001, 0.01]
   - `grad_lr`: 梯度下降学习率，建议范围 [0.001, 0.1]

2. **模式选择**：
   - 快速测试：使用 `dc` 模式
   - 最佳性能：使用 `st` 模式 + `tran` 设置
   - 计算资源有限：使用 `dc` 模式 + `in` 设置

3. **数据集特定设置**：
   - CUB数据集：会自动调整语义特征维度为312
   - 其他数据集：默认使用300维语义特征

## 故障排除

### 1. 导入错误
如果遇到导入错误，确保：
```bash
pip install pulp scikit-learn
```

### 2. CUDA内存不足
对于大型网络，可以：
- 减少 batch size
- 使用 `mode: dc` 减少内存使用
- 设置 `setting: in` 避免转导推理

### 3. 性能问题
如果性能不如预期：
- 检查 `classifier_method` 设置
- 尝试 `setting: tran` 进行转导学习
- 调整学习率参数

## 文件结构

添加SIFT后的相关文件：
```
LibFewShot/
├── core/model/metric/
│   ├── sift.py                    # SIFT模型实现
│   └── __init__.py               # 更新的导入文件
├── config/
│   ├── sift.yaml                 # SIFT基础配置
│   └── classifiers/SIFT.yaml     # SIFT分类器配置
├── test_sift.py                  # 测试脚本
├── requirements.txt              # 更新的依赖
└── SIFT_INTEGRATION.md           # 本文档
```

## 扩展和自定义

要进一步自定义SIFT模型：

1. **修改语义特征维度**：
   在 `core/model/metric/sift.py` 中修改 `z_sem` 参数

2. **添加新的损失函数**：
   在 `compactness_loss` 函数中添加自定义损失

3. **支持新的分类器**：
   在 `classifier_method` 中添加新的选项

4. **数据集特定优化**：
   根据数据集特性调整模型参数

## 参考文献

请引用原始SIFT论文和LibFewShot框架的相关文献。
