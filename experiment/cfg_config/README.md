# 实验配置文件目录

本目录用于存放训练实验的配置文件。每个实验对应一个JSON格式的配置文件。

## 配置文件格式

配置文件应为JSON格式，包含所有训练参数。以下是可用的配置参数：

### 必需参数
- `name`: 实验名称（字符串）

### 可选参数

#### 基本参数
- `output_dir`: 输出目录路径（字符串，默认: "./output"）
- `dataset`: 数据集名称（字符串，默认: "nih"）
  - 可选值: "nih", "pc", "chex", "google", "mimic_ch", "openi", "rsna", "siim", "vin"
  - 可以使用组合，如: "pc-nih-rsna-siim-vin"
- `dataset_dir`: 数据集目录路径（字符串）
- `model`: 模型名称（字符串，默认: "resnet50"）
  - 可选值: "resnet50", "resnet101", "densenet121", "shufflenet_v2_x2_0", "squeezenet1_1"
- `seed`: 随机种子（整数，默认: 0）
- `cuda`: 是否使用CUDA（布尔值，默认: true）

#### 训练参数
- `num_epochs`: 训练轮数（整数，默认: 400）
- `batch_size`: 批次大小（整数，默认: 64）
- `shuffle`: 是否打乱数据（布尔值，默认: true）
- `lr`: 学习率（浮点数，默认: 0.001）
- `threads`: 数据加载线程数（整数，默认: 4）

#### 损失函数参数
- `taskweights`: 是否使用任务权重（布尔值，默认: true）
- `featurereg`: 是否使用特征正则化（布尔值，默认: false）
- `weightreg`: 是否使用权重正则化（布尔值，默认: false）

#### 数据增强参数
- `data_aug`: 是否启用数据增强（布尔值，默认: true）
- `data_aug_rot`: 旋转角度范围（整数，默认: 45）
- `data_aug_trans`: 平移范围（浮点数，默认: 0.15）
- `data_aug_scale`: 缩放范围（浮点数，默认: 0.15）

#### 标签参数
- `label_concat`: 是否连接标签（布尔值，默认: false）
- `label_concat_reg`: 标签连接正则化（布尔值，默认: false）
- `labelunion`: 是否使用标签联合（布尔值，默认: false）

## 使用示例

### 1. 创建配置文件

创建一个新的JSON配置文件，例如 `my_experiment.json`:

```json
{
    "name": "my_experiment",
    "model": "resnet50",
    "dataset": "nih",
    "num_epochs": 100,
    "batch_size": 32,
    "lr": 0.001
}
```

### 2. 启动训练

```bash
./train_manager.sh start experiment_cfg/my_experiment.json
```

或者：

```bash
./train_manager.sh start my_experiment.json
```

### 3. 查看状态

```bash
./train_manager.sh status
```

### 4. 查看日志

```bash
./train_manager.sh logs my_experiment
```

### 5. 停止训练

```bash
./train_manager.sh stop my_experiment
```

## 注意事项

1. 配置文件必须是有效的JSON格式
2. 任务名称可以从配置文件名提取，或使用配置文件中的 `name` 字段
3. 如果某个参数未在配置文件中指定，将使用训练脚本的默认值
4. 布尔值在JSON中使用 `true`/`false`（小写）
