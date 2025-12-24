# 模型测试指南

本指南说明如何测试训练后的模型（包括DenseNet、ResNet和ViT骨干网络）。

## 目录

1. [运行单元测试](#运行单元测试)
2. [测试训练后的模型](#测试训练后的模型)
3. [示例命令](#示例命令)

---

## 运行单元测试

### 运行所有测试

```bash
cd /workspace/torchxrayvision
python3 -m pytest tests/test_models.py -v
```

### 运行特定测试

```bash
# 测试基本模型创建
python3 -m pytest tests/test_models.py::test_model_basic -v

# 测试预训练权重加载
python3 -m pytest tests/test_models.py::test_model_pretrained -v

# 测试模型功能（前向传播、梯度）
python3 -m pytest tests/test_models.py::test_model_function -v

# 测试归一化检查
python3 -m pytest tests/test_models.py::test_normalization_check -v
```

### 运行所有测试（包括其他测试文件）

```bash
python3 -m pytest tests/ -v
```

---

## 测试训练后的模型

我们提供了一个专门的测试脚本 `test_trained_model.py` 来测试训练后的模型checkpoint。

### 基本用法

#### 1. 测试单个模型（单张图像）

```bash
cd /workspace/torchxrayvision/experiment

python3 test_trained_model.py \
    --model_path ./output/nih-vit_b_16-vit_example/nih-vit_b_16-vit_example-best.pt \
    --model_type vit_b_16 \
    --dataset nih \
    --test_image /path/to/test/image.jpg \
    --device cuda
```

#### 2. 测试单个模型（整个测试集）

```bash
# 方式1：指定基础目录，脚本会自动构建完整路径
python3 test_trained_model.py \
    --model_path ./output/nih-vit_b_16-vit_example/nih-vit_b_16-vit_example-best.pt \
    --model_type vit_b_16 \
    --dataset nih \
    --test_dataset_dir /workspace/data \
    --device cuda \
    --batch_size 32

# 方式2：直接指定完整的图像目录路径
python3 test_trained_model.py \
    --model_path ./output/nih-vit_b_16-vit_example/nih-vit_b_16-vit_example-best.pt \
    --model_type vit_b_16 \
    --dataset nih \
    --test_dataset_dir /workspace/data/images-512-NIH \
    --device cuda \
    --batch_size 32
```

#### 3. 比较多个模型

```bash
python3 test_trained_model.py \
    --model_paths ./output/nih-densenet121-model1-best.pt,./output/nih-vit_b_16-model2-best.pt \
    --model_types densenet121,vit_b_16 \
    --dataset nih \
    --test_dataset_dir /workspace/data/NIH/images \
    --device cuda
```

---

## 参数说明

### 必需参数

- `--dataset`: 数据集名称（如 `nih`, `pc`, `mimic_ch` 等）

### 模型路径（二选一）

- `--model_path`: 单个模型checkpoint路径
- `--model_paths`: 多个模型路径，用逗号分隔

### 模型类型

- `--model_type`: 单个模型类型（如 `densenet121`, `resnet50`, `vit_b_16`）
- `--model_types`: 多个模型类型，用逗号分隔

### 测试数据（至少指定一个）

- `--test_image`: 单张测试图像路径
- `--test_dataset_dir`: 测试数据集目录路径
  - 可以指定基础目录（如 `/workspace/data`），脚本会自动根据数据集类型构建完整路径
  - 也可以直接指定完整的图像目录路径（如 `/workspace/data/images-512-NIH`）
  - 脚本会自动检测路径是否正确，如果路径不存在会尝试自动构建
- `--dataset_dir`: 数据集基础目录（可选，如果不指定会从配置文件读取，默认为 `/workspace/data`）

### 可选参数

- `--device`: 计算设备（`cpu` 或 `cuda`，默认：`cpu`）
- `--batch_size`: batch大小（默认：32）
- `--num_classes`: 类别数量（可选，通常会自动推断）

---

## 示例命令

### 示例1：测试ViT模型在单张图像上

```bash
python3 test_trained_model.py \
    --model_path ./output/nih-vit_b_16-vit_example/nih-vit_b_16-vit_example-best.pt \
    --model_type vit_b_16 \
    --dataset nih \
    --test_image /workspace/torchxrayvision/tests/16747_3_1.jpg \
    --device cuda
```

**输出示例：**
```
============================================================
加载模型: ./output/nih-vit_b_16-vit_example/nih-vit_b_16-vit_example-best.pt
模型类型: vit_b_16
============================================================
检测到checkpoint为模型对象
模型加载成功！类别数: 14

测试图像: /workspace/torchxrayvision/tests/16747_3_1.jpg

预测结果:
  Atelectasis: 0.3245
  Cardiomegaly: 0.3567
  Consolidation: 0.4123
  ...
```

### 示例2：测试DenseNet模型在整个测试集上

```bash
python3 test_trained_model.py \
    --model_path ./output/nih-densenet121-densenet_test/nih-densenet121-densenet_test-best.pt \
    --model_type densenet121 \
    --dataset nih \
    --test_dataset_dir /workspace/data/NIH/images \
    --device cuda \
    --batch_size 64
```

### 示例3：比较DenseNet和ViT模型

```bash
python3 test_trained_model.py \
    --model_paths \
        ./output/nih-densenet121-densenet_test/nih-densenet121-densenet_test-best.pt,\
        ./output/nih-vit_b_16-vit_example/nih-vit_b_16-vit_example-best.pt \
    --model_types densenet121,vit_b_16 \
    --dataset nih \
    --test_dataset_dir /workspace/data/NIH/images \
    --device cuda
```

---

## 注意事项

1. **模型类型识别**：脚本会尝试自动识别模型类型，但最好明确指定 `--model_type`

2. **类别数量**：脚本会自动从checkpoint推断类别数量，如果失败可以手动指定 `--num_classes`

3. **数据集路径**：确保测试数据集路径正确，且格式符合 `torchxrayvision` 的要求

4. **设备选择**：如果有GPU，建议使用 `--device cuda` 以加速推理

5. **内存使用**：batch size越大，内存使用越多，根据你的GPU/CPU内存调整

---

## 故障排除

### 问题1：模型加载失败

**错误信息：**
```
错误：加载模型失败: ...
```

**解决方案：**
- 检查模型路径是否正确
- 确认模型类型 `--model_type` 与实际训练的模型一致
- 尝试指定 `--num_classes` 参数

### 问题2：CUDA内存不足

**解决方案：**
- 减小 `--batch_size`（如改为16或8）
- 使用 `--device cpu`

### 问题3：数据集加载失败

**错误信息：**
```
警告：无法使用标准数据集加载器
```

**解决方案：**
- 确认数据集路径正确
- 检查数据集格式是否符合要求
- 可能需要修改脚本中的数据集加载逻辑以适应你的数据格式

---

## 使用预训练模型进行快速测试

如果你想快速测试预训练模型（而非训练后的模型），可以使用项目提供的脚本：

```bash
cd /workspace/torchxrayvision/scripts

# 测试单张图像
python3 process_image.py ../tests/16747_3_1.jpg -resize -weights densenet121-res224-all

# 测试ViT预训练模型
python3 process_image.py ../tests/16747_3_1.jpg -resize -weights vit-base-res224-imagenet
```

---

## 其他测试工具

项目还提供了其他测试工具：

1. **process_batch.py**: 批量处理图像目录
2. **model_calibrate.py**: 模型校准
3. Jupyter notebooks: 在 `scripts/` 目录下有多个示例notebook

详情请参考 `scripts/README.md`。

