import os
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join

import numpy as np
import torch
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn, sklearn.model_selection
import torchxrayvision as xrv

from tqdm import tqdm as tqdm_base
import sys

def tqdm(*args, **kwargs):
    # 检测是否在非交互式环境（如日志文件）中运行
    # 如果是，则完全禁用进度条以减少日志输出
    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    # 如果输出被重定向到文件，完全禁用进度条
    if not is_tty:
        kwargs.setdefault('disable', True)  # 完全禁用进度条
    else:
        # 交互式环境中使用默认设置
        pass
    
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)
#from tqdm.auto import tqdm




def train(model, dataset, cfg):
    print("Our config:")
    pprint.pprint(cfg)
        
    dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
    
    device = 'cuda' if cfg.cuda else 'cpu'
    if not torch.cuda.is_available() and cfg.cuda:
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')

    print(cfg.output_dir)

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    
    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Dataset    
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    valid_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    #print(model)

    # Optimizer
    optimizer_type = getattr(cfg, 'optimizer', 'adam').lower()
    weight_decay = getattr(cfg, 'weight_decay', 1e-5)
    
    if optimizer_type == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    else:
        # Default to Adam (backward compatibility)
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay, amsgrad=True)
    print(f"Optimizer: {optim}")
    
    # Learning rate scheduler
    scheduler = None
    scheduler_type = getattr(cfg, 'scheduler', None)
    warmup_ratio = getattr(cfg, 'warmup_ratio', 0.0)
    
    if scheduler_type == 'cosine':
        # Cosine annealing scheduler
        # T_max is the number of epochs after warmup
        warmup_epochs = int(cfg.num_epochs * warmup_ratio) if warmup_ratio > 0 else 0
        T_max = cfg.num_epochs - warmup_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=T_max, eta_min=0)
        print(f"Scheduler: CosineAnnealingLR (T_max={T_max}, warmup_epochs={warmup_epochs})")
    elif warmup_ratio > 0:
        # Only warmup without scheduler
        warmup_epochs = int(cfg.num_epochs * warmup_ratio)
        print(f"Warmup: {warmup_epochs} epochs ({warmup_ratio*100:.1f}% of total)")

    # ========================================================================
    # Loss 函数定义位置
    # ========================================================================
    # Loss 函数: torch.nn.BCEWithLogitsLoss()
    # - 功能: 二元交叉熵损失（Binary Cross Entropy with Logits）
    # - 适用场景: 多标签分类任务
    # - 特点: 内部结合了 sigmoid 和 BCE，数值稳定性更好
    # - 输入: model 输出的 logits (未经过 sigmoid) 和真实标签
    # - 注意: 在 train_epoch() 中会对每个任务(task)分别计算 loss
    # ========================================================================
    criterion = torch.nn.BCEWithLogitsLoss()

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(join(cfg.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(cfg.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)
    
    # Calculate warmup epochs
    warmup_ratio = getattr(cfg, 'warmup_ratio', 0.0)
    warmup_epochs = int(cfg.num_epochs * warmup_ratio) if warmup_ratio > 0 else 0
    initial_lr = cfg.lr
    
    for epoch in range(start_epoch, cfg.num_epochs):
        
        # Warmup learning rate
        if epoch < warmup_epochs and warmup_epochs > 0:
            # Linear warmup
            current_lr = initial_lr * (epoch + 1) / warmup_epochs
            for param_group in optim.param_groups:
                param_group['lr'] = current_lr
        elif epoch == warmup_epochs and warmup_epochs > 0:
            # Ensure lr is set to initial_lr after warmup before scheduler starts
            for param_group in optim.param_groups:
                param_group['lr'] = initial_lr
        
        avg_loss = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion)
        
        auc_valid = valid_test_epoch(name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion=criterion)[0]
        
        # Update learning rate scheduler (after warmup)
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()
        
        # Print learning rate info periodically
        if epoch % 10 == 0 or epoch == cfg.num_epochs - 1:
            current_lr = optim.param_groups[0]['lr']
            if epoch < warmup_epochs:
                print(f"Epoch {epoch+1}/{cfg.num_epochs}: Learning rate (warmup) = {current_lr:.6f}")
            else:
                print(f"Epoch {epoch+1}/{cfg.num_epochs}: Learning rate = {current_lr:.6f}")

        if np.mean(auc_valid) > best_metric:
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best.pt'))
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        torch.save(model, join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

    return metrics, best_metric, weights_for_best_validauc





def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, limit=None):
    model.train()

    if cfg.taskweights:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights/weights.max()
        weights = torch.from_numpy(weights).to(device).float()
        print("task weights", weights)
    
    avg_loss = []
    # 检测是否在非交互式环境
    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    t = tqdm(train_loader)
    
    # 在非交互式环境中，定期打印进度而不是使用进度条
    log_interval = 50  # 每50个iteration打印一次
    last_log_time = 0
    
    for batch_idx, samples in enumerate(t):
        
        if limit and (batch_idx > limit):
            print("breaking out")
            break
            
        optimizer.zero_grad()
        
        images = samples["img"].float().to(device)
        targets = samples["lab"].to(device)

        outputs = model(images)
        
        loss = torch.zeros(1).to(device).float()
        for task in range(targets.shape[1]):
            task_output = outputs[:,task]
            task_target = targets[:,task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = criterion(task_output.float(), task_target.float())
                if cfg.taskweights:
                    loss += weights[task]*task_loss
                else:
                    loss += task_loss
        
        # here regularize the weight matrix when label_concat is used
        if cfg.label_concat_reg:
            if not cfg.label_concat:
                raise Exception("cfg.label_concat must be true")
            weight = model.classifier.weight
            num_labels = len(xrv.datasets.default_pathologies)
            num_datasets = weight.shape[0]//num_labels
            weight_stacked = weight.reshape(num_datasets,num_labels,-1)
            label_concat_reg_lambda = torch.tensor(0.1).to(device).float()
            for task in range(num_labels):
                dists = torch.pdist(weight_stacked[:,task], p=2).mean()
                loss += label_concat_reg_lambda*dists
                
        loss = loss.sum()
        
        if cfg.featurereg:
            feat = model.features(images)
            loss += feat.abs().sum()
            
        if cfg.weightreg:
            loss += model.classifier.weight.abs().sum()
        
        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        
        # 在交互式环境中，正常更新进度条描述
        if is_tty:
            t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')
        else:
            # 在非交互式环境中，定期打印简洁的进度信息
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                import time
                current_time = time.time()
                elapsed = current_time - last_log_time if last_log_time > 0 else 0
                last_log_time = current_time
                progress = 100.0 * (batch_idx + 1) / len(train_loader)
                print(f'Epoch {epoch + 1} - Train [{batch_idx + 1}/{len(train_loader)} ({progress:.1f}%)] - Loss = {np.mean(avg_loss):4.4f}')

        optimizer.step()

    return np.mean(avg_loss)

def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        t = tqdm(data_loader)
        
        # 在非交互式环境中，定期打印进度而不是使用进度条
        log_interval = 50  # 每50个iteration打印一次
        
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)

            outputs = model(images)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            
            # 在交互式环境中，正常更新进度条描述
            if is_tty:
                t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            else:
                # 在非交互式环境中，定期打印简洁的进度信息
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(data_loader):
                    progress = 100.0 * (batch_idx + 1) / len(data_loader)
                    print(f'Epoch {epoch + 1} - {name} [{batch_idx + 1}/{len(data_loader)} ({progress:.1f}%)] - Loss = {np.mean(avg_loss):4.4f}')
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        # ====================================================================
        # Metrics (AUC) 计算位置
        # ====================================================================
        # AUC 计算: sklearn.metrics.roc_auc_score()
        # - 功能: 计算每个任务(类别)的 ROC-AUC 值
        # - 输入: task_targets[task] (真实标签), task_outputs[task] (模型输出的 logits)
        # - 输出: 每个任务的 AUC 值
        # - 最终指标: 所有任务 AUC 的平均值 (macro-averaged AUC)
        # - 注意: 如果某个任务只有单一类别，则返回 NaN
        # ====================================================================
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    return auc, task_aucs, task_outputs, task_targets
