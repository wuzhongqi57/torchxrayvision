#!/usr/bin/env python
# coding: utf-8

# ============================================================================
# NIH ChestX-ray 多标签训练主入口脚本
# ============================================================================
# 本脚本是 NIH ChestX-ray 多标签分类训练的主入口文件
# 主要功能：数据加载、模型创建、训练流程调用
# ============================================================================

import os,sys
sys.path.insert(0,"..")
import os,sys,inspect
from glob import glob
from os.path import exists, join
# import matplotlib.pyplot as plt  # Not used in training script
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection

import random
import train_utils
import torchxrayvision as xrv


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('-name', type=str)
parser.add_argument('--output_dir', type=str, default="/scratch/users/joecohen/output/")
parser.add_argument('--dataset', type=str, default="pc-nih-rsna-siim-vin")
parser.add_argument('--dataset_dir', type=str, default="/home/groups/akshaysc/joecohen/")
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=400, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer type: adam or adamw')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'cosine'], help='Learning rate scheduler: None or cosine')
parser.add_argument('--warmup_ratio', type=float, default=0.0, help='Warmup ratio (e.g., 0.05 for 5%% of epochs)')
parser.add_argument('--threads', type=int, default=4, help='')
parser.add_argument('--taskweights', type=bool, default=True, help='')
parser.add_argument('--featurereg', type=bool, default=False, help='')
parser.add_argument('--weightreg', type=bool, default=False, help='')
parser.add_argument('--data_aug', type=bool, default=True, help='')
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
parser.add_argument('--label_concat', type=bool, default=False, help='')
parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
parser.add_argument('--labelunion', type=bool, default=False, help='')

cfg = parser.parse_args()
print(cfg)

data_aug = None
if cfg.data_aug:
    data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                            translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                            scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
        torchvision.transforms.ToTensor()
    ])
    print(data_aug)

# Determine input resolution based on model type
# ViT models typically use 224x224, while other models use 512x512
input_resolution = 224 if "vit" in cfg.model.lower() else 512
transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(input_resolution)])
print(f"Using input resolution: {input_resolution}x{input_resolution} for model: {cfg.model}")

datas = []
datas_names = []
if "nih" in cfg.dataset:
    # ========================================================================
    # Dataset 类定义位置
    # ========================================================================
    # 使用的 Dataset 类：xrv.datasets.NIH_Dataset
    # 类定义文件路径：torchxrayvision/datasets.py (第383行开始)
    # 功能：加载 NIH ChestX-ray14 数据集，包含14种胸部疾病的 multi-label 标注
    # 返回格式：{"img": 图像张量, "lab": 标签向量(14维)}
    # ========================================================================
    dataset = xrv.datasets.NIH_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-NIH", 
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("nih")
if "pc" in cfg.dataset:
    dataset = xrv.datasets.PC_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-PC", 
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("pc")
if "chex" in cfg.dataset:
    dataset = xrv.datasets.CheX_Dataset(
        imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
        csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)
    datas.append(dataset)
    datas_names.append("chex")
if "google" in cfg.dataset:
    dataset = xrv.datasets.NIH_Google_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-NIH",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("google")
if "mimic_ch" in cfg.dataset:
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath="/scratch/users/joecohen/data/MIMICCXR-2.0/files/",
        csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
        metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("mimic_ch")
if "openi" in cfg.dataset:
    dataset = xrv.datasets.Openi_Dataset(
        imgpath=cfg.dataset_dir + "/OpenI/images/",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("openi")
if "rsna" in cfg.dataset:
    dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("rsna")
if "siim" in cfg.dataset:
    dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
        imgpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/dicom-images-train",
        csvpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/train-rle.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("siim")
if "vin" in cfg.dataset:
    dataset = xrv.datasets.VinBrain_Dataset(
        imgpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train",
        csvpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train.csv",
        transform=transform, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("vin")


print("datas_names", datas_names)

if cfg.labelunion:
    newlabels = set()
    for d in datas:
        newlabels = newlabels.union(d.pathologies)
    newlabels.remove("Support Devices")
    print(list(newlabels))
    for d in datas:
        xrv.datasets.relabel_dataset(list(newlabels), d)
else:
    for d in datas:
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d)

#cut out training sets
train_datas = []
test_datas = []
for i, dataset in enumerate(datas):
    
    # give patientid if not exist
    if "patientid" not in dataset.csv:
        dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
        
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
    
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
    
    train_datas.append(train_dataset)
    test_datas.append(test_dataset)
    
if len(datas) == 0:
    raise Exception("no dataset")
elif len(datas) == 1:
    train_dataset = train_datas[0]
    test_dataset = test_datas[0]
else:
    print("merge datasets")
    train_dataset = xrv.datasets.Merge_Dataset(train_datas)
    test_dataset = xrv.datasets.Merge_Dataset(test_datas)


# Setting the seed
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("train_dataset.labels.shape", train_dataset.labels.shape)
print("test_dataset.labels.shape", test_dataset.labels.shape)
print("train_dataset",train_dataset)
print("test_dataset",test_dataset)
    
# ============================================================================
# 模型 Backbone 和分类 Head 定义位置
# ============================================================================
# create models
if "densenet" in cfg.model:
    model = xrv.models.DenseNet(num_classes=train_dataset.labels.shape[1], in_channels=1, 
                                **xrv.models.get_densenet_params(cfg.model)) 
elif "resnet101" in cfg.model:
    model = torchvision.models.resnet101(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
elif "resnet50" in cfg.model:
    # ========================================================================
    # 模型 Backbone 定义位置
    # ========================================================================
    # Backbone: torchvision.models.resnet50
    # - 使用 ResNet50 作为特征提取 backbone
    # - 修改第一层卷积：从 3 通道(RGB) 改为 1 通道(灰度X光片)
    # - Backbone 结构：conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool
    # ========================================================================
    model = torchvision.models.resnet50(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # ========================================================================
    # 分类 Head 结构
    # ========================================================================
    # 分类 Head: model.fc (全连接层)
    # - 输入维度: 2048 (ResNet50 backbone 输出的特征维度)
    # - 输出维度: train_dataset.labels.shape[1] (多标签类别数，通常为14)
    # - 结构: Linear(2048 -> num_classes)
    # - 输出: 每个类别的 logits (未经过 sigmoid)
    # - 注意: 由于是多标签分类，每个类别独立输出，不使用 softmax
    # ========================================================================
    
elif "vit" in cfg.model.lower():
    # ========================================================================
    # ViT (Vision Transformer) 模型定义位置
    # ========================================================================
    # Backbone: torchvision.models.vit_b_16 (通过 xrv.models.ViT 封装)
    # - 使用 Vision Transformer 作为特征提取 backbone
    # - 自动适配单通道输入（X射线图像）
    # - 支持 ImageNet 预训练权重（可选）
    # - 输入分辨率: 224x224 (ViT 标准输入尺寸)
    # - Backbone 结构: patch embedding -> transformer encoder -> classification head
    # ========================================================================
    # 创建 ViT 模型
    # - weights=None: 不使用预训练的任务特定权重（从头训练）
    # - use_imagenet_pretrained: 是否使用 ImageNet 预训练的 backbone
    #   设置为 True 可以使用 ImageNet 预训练权重进行迁移学习
    # - num_classes 通过修改分类头自动设置
    # ========================================================================
    use_imagenet_pretrained = True  # 使用 ImageNet 预训练权重进行迁移学习
    model = xrv.models.ViT(
        weights=None,  # 不使用任务特定权重，从头训练分类头
        use_imagenet_pretrained=use_imagenet_pretrained,
        op_threshs=None
    )
    # 替换分类头以匹配数据集类别数
    model.backbone.heads = torch.nn.Linear(model.hidden_dim, train_dataset.labels.shape[1])
    # 为了兼容 train_utils.py 中的 model.classifier 引用，添加 classifier 属性
    # train_utils.py 在 label_concat_reg 和 weightreg 中会使用 model.classifier
    model.classifier = model.backbone.heads
    print(f"ViT model created with ImageNet pretrained backbone: {use_imagenet_pretrained}")
    print(f"ViT hidden dimension: {model.hidden_dim}")
    print(f"ViT classification head: {train_dataset.labels.shape[1]} classes")
    
elif "shufflenet_v2_x2_0" in cfg.model:
    model = torchvision.models.shufflenet_v2_x2_0(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
elif "squeezenet1_1" in cfg.model:
    model = torchvision.models.squeezenet1_1(num_classes=train_dataset.labels.shape[1], pretrained=False)
    #patch for single channel
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
else:
    raise Exception("no model")

# ============================================================================
# 训练流程入口
# ============================================================================
# 调用 train_utils.train() 开始训练
# 训练流程包括：loss 计算、metrics (AUC) 计算等，详见 train_utils.py
# ============================================================================
train_utils.train(model, train_dataset, cfg)


print("Done")
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                                            batch_size=cfg.batch_size,
#                                            shuffle=cfg.shuffle,
#                                            num_workers=0, pin_memory=False)


# ============================================================================
# 完整前向流程总结：数据 → 模型 → loss → metric
# ============================================================================
# 
# 1. 数据加载 (Data Loading)
#    - 使用 NIH_Dataset 类 (torchxrayvision/datasets.py:383) 加载 NIH ChestX-ray 数据
#    - 数据预处理：XRayCenterCrop() + XRayResizer(512) + 数据增强(可选)
#    - 输出格式：{"img": [B, 1, 512, 512], "lab": [B, 14]} (B=batch_size, 14=多标签类别数)
# 
# 2. 模型前向传播 (Model Forward)
#    - Backbone: ResNet50 (torchvision.models.resnet50)
#      * 输入: [B, 1, 512, 512] 灰度X光图像
#      * 第一层修改: conv1 从 3 通道改为 1 通道
#      * 特征提取: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool
#      * 输出特征: [B, 2048]
#    - 分类 Head: model.fc (全连接层)
#      * 输入: [B, 2048] (backbone 输出的特征向量)
#      * 输出: [B, 14] (每个类别的 logits，未经过 sigmoid)
# 
# 3. Loss 计算 (Loss Computation)
#    - Loss 函数: BCEWithLogitsLoss() (train_utils.py:77)
#    - 计算方式: 对每个任务(类别)分别计算 loss，然后求和
#      * 对每个 task: loss_task = BCEWithLogitsLoss(logits[:, task], labels[:, task])
#      * 总 loss = sum(loss_task) (可选：使用 taskweights 加权)
#    - 特点: 支持 NaN 标签（缺失标注），只对有效标签计算 loss
# 
# 4. Metrics 计算 (AUC Metric)
#    - 计算位置: train_utils.py:266 (valid_test_epoch 函数中)
#    - 计算方式:
#      * 收集所有 batch 的预测 logits 和真实标签
#      * 对每个任务: task_auc = roc_auc_score(task_targets, task_outputs)
#      * 最终指标: mean(task_aucs) (macro-averaged AUC)
#    - 注意: 如果某个任务只有单一类别，则跳过该任务的 AUC 计算
# 
# 完整流程示例:
#   数据 → NIH_Dataset.__getitem__() → {"img": tensor, "lab": array}
#        → DataLoader → batch {"img": [B,1,512,512], "lab": [B,14]}
#        → model(images) → ResNet50 backbone → [B,2048] features
#                        → model.fc → [B,14] logits
#        → BCEWithLogitsLoss(logits, labels) → scalar loss
#        → loss.backward() → optimizer.step() (训练时)
#        → roc_auc_score(targets, outputs) → AUC metrics (验证时)
# 
# ============================================================================


