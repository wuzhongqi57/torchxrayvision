#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试训练后的模型脚本

用途：加载训练好的模型checkpoint并进行测试
支持：DenseNet, ResNet, ViT等所有骨干网络

使用方法：
    # 测试单个模型
    python test_trained_model.py --model_path ./output/nih-vit_b_16-vit_example/nih-vit_b_16-vit_example-best.pt \
                                  --model_type vit_b_16 \
                                  --dataset nih \
                                  --test_dataset_dir /workspace/data/NIH/images

    # 测试并比较多个模型
    python test_trained_model.py --model_paths ./output/nih-densenet121-best.pt,./output/nih-vit_b_16-best.pt \
                                  --model_types densenet121,vit_b_16 \
                                  --dataset nih \
                                  --test_dataset_dir /workspace/data/NIH/images
"""

import os
import sys
import argparse
import torch
import numpy as np
from glob import glob
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torchxrayvision as xrv
from scripts import train_utils


def load_trained_model(model_path, model_type, dataset_name, num_classes=None):
    """
    加载训练后的模型
    
    Args:
        model_path: 模型checkpoint路径 (.pt文件)
        model_type: 模型类型 ('densenet121', 'resnet50', 'vit_b_16'等)
        dataset_name: 数据集名称 (用于确定num_classes)
        num_classes: 类别数量 (可选，如果不提供则从checkpoint推断)
    
    Returns:
        model: 加载好的模型
    """
    print(f"\n{'='*60}")
    print(f"加载模型: {model_path}")
    print(f"模型类型: {model_type}")
    print(f"{'='*60}")
    
    # 首先加载checkpoint以确定模型结构
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 判断checkpoint格式
    if isinstance(checkpoint, torch.nn.Module):
        # 直接是模型对象（train_utils.py保存的格式）
        print("检测到checkpoint为模型对象")
        model = checkpoint
        # 尝试从模型获取num_classes
        if num_classes is None:
            if hasattr(model, 'classifier'):
                num_classes = model.classifier.out_features if hasattr(model.classifier, 'out_features') else model.classifier.weight.shape[0]
            elif hasattr(model, 'backbone') and hasattr(model.backbone, 'heads'):
                num_classes = model.backbone.heads.out_features
            elif hasattr(model, 'fc'):
                num_classes = model.fc.out_features if hasattr(model.fc, 'out_features') else model.fc.weight.shape[0]
            else:
                # 尝试从输出维度推断
                dummy_input = torch.randn(1, 1, 224, 224)
                with torch.no_grad():
                    output = model(dummy_input)
                    num_classes = output.shape[1]
    else:
        # 是字典格式，需要创建模型
        print("检测到checkpoint为字典格式")
        if num_classes is None:
            # 尝试从state_dict推断
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 尝试找到分类头的输出维度
            for key in state_dict.keys():
                if 'classifier' in key or 'heads' in key or 'fc' in key:
                    if 'weight' in key:
                        num_classes = state_dict[key].shape[0]
                        break
        
        # 创建模型
        if 'densenet' in model_type.lower():
            model = xrv.models.DenseNet(num_classes=num_classes or 14, in_channels=1)
        elif 'resnet' in model_type.lower():
            import torchvision
            if 'resnet50' in model_type.lower():
                model = torchvision.models.resnet50(num_classes=num_classes or 14, pretrained=False)
                model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            elif 'resnet101' in model_type.lower():
                model = torchvision.models.resnet101(num_classes=num_classes or 14, pretrained=False)
                model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            else:
                raise ValueError(f"不支持的ResNet变体: {model_type}")
        elif 'vit' in model_type.lower():
            model = xrv.models.ViT(weights=None, use_imagenet_pretrained=False)
            # 如果需要，修改分类头
            if num_classes is not None:
                model.backbone.heads = torch.nn.Linear(model.hidden_dim, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载权重
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    print(f"模型加载成功！类别数: {num_classes}")
    
    # 设置模型为评估模式
    model.eval()
    
    return model


def get_dataset_path(dataset_name, dataset_dir=None):
    """
    根据数据集名称和基础目录构建正确的数据集路径
    
    Args:
        dataset_name: 数据集名称 (nih, pc, chex等)
        dataset_dir: 数据集基础目录（可选，默认从配置文件读取或使用/workspace/data）
    
    Returns:
        imgpath: 图像目录路径
    """
    # 如果dataset_dir未指定，尝试从配置文件或使用默认值
    if dataset_dir is None:
        # 尝试从配置文件读取
        import json
        cfg_dir = Path(__file__).parent / "cfg_config"
        default_dataset_dir = "/workspace/data"
        
        # 查找配置文件中的dataset_dir
        if cfg_dir.exists():
            for cfg_file in cfg_dir.glob("*.json"):
                try:
                    with open(cfg_file) as f:
                        config = json.load(f)
                        if 'dataset_dir' in config:
                            default_dataset_dir = config['dataset_dir']
                            break
                except:
                    continue
        
        dataset_dir = default_dataset_dir
    
    # 根据数据集类型构建路径（与train_model.py保持一致）
    dataset_name_lower = dataset_name.lower()
    
    if 'nih' in dataset_name_lower:
        imgpath = os.path.join(dataset_dir, "images-512-NIH")
    elif 'pc' in dataset_name_lower:
        imgpath = os.path.join(dataset_dir, "images-512-PC")
    elif 'chex' in dataset_name_lower:
        imgpath = os.path.join(dataset_dir, "CheXpert-v1.0-small")
    elif 'rsna' in dataset_name_lower:
        imgpath = os.path.join(dataset_dir, "kaggle-pneumonia-jpg/stage_2_train_images_jpg")
    elif 'siim' in dataset_name_lower:
        imgpath = os.path.join(dataset_dir, "SIIM_TRAIN_TEST/dicom-images-train")
    elif 'mimic' in dataset_name_lower:
        imgpath = "/scratch/users/joecohen/data/MIMICCXR-2.0/files/"  # MIMIC有固定路径
    elif 'openi' in dataset_name_lower:
        imgpath = os.path.join(dataset_dir, "OpenI/images")
    else:
        # 如果无法识别，直接使用dataset_dir
        imgpath = dataset_dir
    
    return imgpath


def evaluate_model(model, test_dataset_dir, dataset_name, device='cpu', batch_size=32):
    """
    在测试集上评估模型
    
    Args:
        model: 模型
        test_dataset_dir: 测试数据集目录（可以是完整路径或基础目录）
        dataset_name: 数据集名称
        device: 设备 ('cpu' 或 'cuda')
        batch_size: batch大小
    
    Returns:
        metrics: 评估指标字典
    """
    print(f"\n{'='*60}")
    print(f"开始评估模型")
    print(f"数据集名称: {dataset_name}")
    print(f"{'='*60}")
    
    # 检查test_dataset_dir是否存在
    if not os.path.exists(test_dataset_dir):
        print(f"警告：指定的路径不存在: {test_dataset_dir}")
        print(f"尝试根据数据集名称自动构建路径...")
        
        # 尝试多种方式构建路径（按优先级排序）
        # 优先尝试标准的基础目录，避免使用错误路径的父目录
        possible_base_dirs = [
            "/workspace/data",  # 最常用的基础目录，优先尝试
        ]
        
        # 如果路径看起来像是错误的子路径（如包含/NIH/或/images），尝试提取到data目录
        path_lower = test_dataset_dir.lower()
        if '/data' in path_lower:
            parts = [p for p in test_dataset_dir.split('/') if p]
            if 'data' in parts:
                data_idx = parts.index('data')
                if data_idx > 0:
                    base_path = '/' + '/'.join(parts[:data_idx+1])
                    if base_path not in possible_base_dirs:
                        possible_base_dirs.append(base_path)
        
        # 最后尝试父目录（可能不正确，但作为fallback）
        parent_dir = os.path.dirname(test_dataset_dir) if os.path.dirname(test_dataset_dir) else None
        if parent_dir and parent_dir not in possible_base_dirs:
            possible_base_dirs.append(parent_dir)
        
        possible_base_dirs = [d for d in possible_base_dirs if d]
        
        suggested_path = None
        for base_dir in possible_base_dirs:
            try_path = get_dataset_path(dataset_name, base_dir)
            if os.path.exists(try_path):
                suggested_path = try_path
                print(f"找到数据集路径: {suggested_path}")
                break
        
        if suggested_path and os.path.exists(suggested_path):
            imgpath = suggested_path
        else:
            # 最后尝试：直接使用数据集名称构建标准路径
            standard_path = get_dataset_path(dataset_name, None)
            if os.path.exists(standard_path):
                print(f"找到数据集路径（从配置文件）: {standard_path}")
                imgpath = standard_path
            else:
                print(f"错误：无法找到数据集路径")
                print(f"  尝试的路径: {suggested_path or standard_path}")
                print(f"  提示：请检查数据集是否正确下载并解压")
                print(f"  对于NIH数据集，应该是: <dataset_dir>/images-512-NIH")
                print(f"  你可以直接指定正确的路径，如: /workspace/data/images-512-NIH")
                print(f"  或者指定基础目录: /workspace/data")
                return None
    else:
        # 如果路径存在，检查是否是完整的图像目录或基础目录
        dataset_name_lower = dataset_name.lower()
        
        # 检查是否是完整的图像目录
        # 判断标准：目录中包含图像文件（.jpg, .png等）
        dir_contents = os.listdir(test_dataset_dir) if os.path.isdir(test_dataset_dir) else []
        has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')) for f in dir_contents[:10]) if dir_contents else False
        
        if has_images:
            # 看起来是图像目录，直接使用
            print(f"检测到图像目录，使用路径: {test_dataset_dir}")
            imgpath = test_dataset_dir
        elif 'nih' in dataset_name_lower and 'images-512-NIH' not in test_dataset_dir:
            # 可能是基础目录，尝试构建完整路径
            suggested_path = get_dataset_path(dataset_name, test_dataset_dir)
            if os.path.exists(suggested_path):
                print(f"检测到基础目录，自动构建路径: {suggested_path}")
                imgpath = suggested_path
            else:
                print(f"警告：路径存在但不是图像目录，尝试直接使用: {test_dataset_dir}")
                imgpath = test_dataset_dir
        else:
            imgpath = test_dataset_dir
    
    print(f"使用数据集路径: {imgpath}")
    
    # 创建测试数据集
    try:
        if dataset_name.lower() == 'nih':
            test_dataset = xrv.datasets.NIH_Dataset(
                imgpath=imgpath,
                transform=xrv.datasets.XRayCenterCrop(),
                unique_patients=False,
                views=["PA", "AP"]
            )
        elif 'pc' in dataset_name.lower():
            test_dataset = xrv.datasets.PC_Dataset(
                imgpath=imgpath,
                transform=xrv.datasets.XRayCenterCrop(),
                unique_patients=False,
                views=["PA", "AP"]
            )
        elif 'chex' in dataset_name.lower():
            # CheXpert需要csvpath
            csvpath = os.path.join(os.path.dirname(imgpath), "train.csv") if os.path.dirname(imgpath) else None
            if not csvpath or not os.path.exists(csvpath):
                csvpath = None  # 使用默认的csvpath
            test_dataset = xrv.datasets.CheX_Dataset(
                imgpath=imgpath,
                csvpath=csvpath,
                transform=xrv.datasets.XRayCenterCrop(),
                unique_patients=False
            )
        else:
            # 通用数据集加载
            print(f"警告：使用通用数据集加载器，可能不完全兼容")
            test_dataset = xrv.datasets.XRayDataset(
                imgpath=imgpath,
                transform=xrv.datasets.XRayCenterCrop()
            )
    except Exception as e:
        print(f"错误：无法加载数据集: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 创建DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    # 将模型移到设备
    model = model.to(device)
    
    # 评估
    criterion = torch.nn.BCEWithLogitsLoss()
    
    metrics = train_utils.valid_test_epoch(
        name='Test',
        epoch=0,
        model=model,
        device=device,
        data_loader=test_loader,
        criterion=criterion
    )
    
    return metrics


def test_single_image(model, image_path, device='cpu'):
    """
    测试单张图像
    
    Args:
        model: 模型
        image_path: 图像路径
        device: 设备
    
    Returns:
        predictions: 预测结果字典
    """
    print(f"\n测试图像: {image_path}")
    
    # 加载图像并进行预处理（与训练时保持一致）
    img = xrv.utils.load_image(image_path)
    
    # 应用预处理transform
    import torchvision
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)  # ViT使用224，其他模型也会自动resize
    ])
    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W] -> 添加batch维度在模型中
    
    # 移到设备
    model = model.to(device)
    img = img.to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        # 应用sigmoid得到概率
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # 获取病理名称（如果模型有pathologies属性）
    if hasattr(model, 'pathologies'):
        pathologies = model.pathologies
    elif hasattr(model, 'targets'):
        pathologies = model.targets
    else:
        # 使用默认的病理名称
        pathologies = xrv.datasets.default_pathologies[:len(probs)]
    
    # 创建预测字典
    predictions = dict(zip(pathologies, probs))
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='测试训练后的模型')
    
    # 模型路径参数
    parser.add_argument('--model_path', type=str, help='单个模型checkpoint路径')
    parser.add_argument('--model_paths', type=str, help='多个模型路径，用逗号分隔')
    
    # 模型类型
    parser.add_argument('--model_type', type=str, help='模型类型 (densenet121, resnet50, vit_b_16等)')
    parser.add_argument('--model_types', type=str, help='多个模型类型，用逗号分隔')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (nih, pc等)')
    parser.add_argument('--test_dataset_dir', type=str, help='测试数据集目录路径（可以是完整路径或基础目录，如/workspace/data）')
    parser.add_argument('--test_image', type=str, help='单张测试图像路径')
    parser.add_argument('--dataset_dir', type=str, default=None, help='数据集基础目录（可选，如果不指定会从配置文件读取）')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--batch_size', type=int, default=32, help='batch大小')
    parser.add_argument('--num_classes', type=int, default=None, help='类别数量（可选）')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.model_path and args.model_paths:
        print("错误：不能同时指定 --model_path 和 --model_paths")
        return
    
    if not args.model_path and not args.model_paths:
        print("错误：必须指定 --model_path 或 --model_paths")
        return
    
    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告：CUDA不可用，使用CPU")
        device = 'cpu'
    
    # 处理单个或多个模型
    if args.model_path:
        model_paths = [args.model_path]
        model_types = [args.model_type] if args.model_type else ['unknown']
    else:
        model_paths = [p.strip() for p in args.model_paths.split(',')]
        if args.model_types:
            model_types = [t.strip() for t in args.model_types.split(',')]
        else:
            model_types = ['unknown'] * len(model_paths)
    
    # 确保模型路径和类型数量匹配
    if len(model_paths) != len(model_types):
        print(f"警告：模型路径数量({len(model_paths)})与模型类型数量({len(model_types)})不匹配")
        model_types = model_types + ['unknown'] * (len(model_paths) - len(model_types))
    
    # 测试每个模型
    results = []
    for model_path, model_type in zip(model_paths, model_types):
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"错误：模型文件不存在: {model_path}")
            continue
        
        # 加载模型
        try:
            model = load_trained_model(
                model_path=model_path,
                model_type=model_type,
                dataset_name=args.dataset,
                num_classes=args.num_classes
            )
        except Exception as e:
            print(f"错误：加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 测试单张图像
        if args.test_image:
            if os.path.exists(args.test_image):
                predictions = test_single_image(model, args.test_image, device)
                print("\n预测结果:")
                for patho, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {patho}: {prob:.4f}")
                results.append({
                    'model_path': model_path,
                    'model_type': model_type,
                    'test_image': args.test_image,
                    'predictions': predictions
                })
            else:
                print(f"错误：测试图像不存在: {args.test_image}")
        
        # 在测试集上评估
        if args.test_dataset_dir:
            # 使用指定的dataset_dir或从参数获取
            test_dataset_dir = args.test_dataset_dir
            if args.dataset_dir:
                # 如果指定了dataset_dir，将其作为基础目录
                test_dataset_dir = args.dataset_dir
            
            try:
                metrics = evaluate_model(
                    model=model,
                    test_dataset_dir=test_dataset_dir,
                    dataset_name=args.dataset,
                    device=device,
                    batch_size=args.batch_size
                )
                if metrics:
                    print(f"\n评估结果:")
                    print(f"  AUC: {metrics[0]:.4f}")
                    if len(metrics) > 1:
                        print(f"  其他指标: {metrics[1:]}")
                    results.append({
                        'model_path': model_path,
                        'model_type': model_type,
                        'metrics': metrics
                    })
            except Exception as e:
                print(f"错误：评估失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 如果有多个模型，进行比较
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("模型比较结果")
        print(f"{'='*60}")
        for i, result in enumerate(results):
            print(f"\n模型 {i+1}: {result['model_path']}")
            if 'metrics' in result:
                print(f"  AUC: {result['metrics'][0]:.4f}")
            if 'predictions' in result:
                print(f"  预测类别数: {len(result['predictions'])}")


if __name__ == '__main__':
    main()
