# import torch
# import numpy as np
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# def safe_divide(numerator, denominator):
#     return numerator / denominator if denominator != 0 else 0
#
#
# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, criterion):
#     model.train()
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#     accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
#     optimizer.zero_grad()
#
#     sample_num = 0
#     all_preds = []
#     all_labels = []
#     data_loader = tqdm(data_loader, file=sys.stdout)
#
#     for step, data in enumerate(data_loader):
#         h_images, v_images, text_data, labels, h_path, v_path = data
#         pred = model(h_images.to(device), v_images.to(device), text_data.to(device))
#
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         loss = criterion(pred, labels.to(device))  # 使用传入的criterion计算损失
#         loss.backward()
#         accu_loss += loss.detach()
#
#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             return None
#
#         optimizer.step()
#         optimizer.zero_grad()
#         lr_scheduler.step()
#
#         sample_num += labels.size(0)
#         all_preds.extend(pred_classes.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
#     # 计算混淆矩阵
#     num_classes = 3
#     cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
#
#     recall = [safe_divide(np.diag(cm)[i], np.sum(cm[i, :])) for i in range(num_classes)]
#     precision = [safe_divide(np.diag(cm)[i], np.sum(cm[:, i])) for i in range(num_classes)]
#
#     specificity = []
#     for i in range(num_classes):
#         TP = cm[i, i]
#         FP = np.sum(cm[:, i]) - TP
#         FN = np.sum(cm[i, :]) - TP
#         TN = np.sum(cm) - TP - FP - FN
#         spec = safe_divide(TN, TN + FP)
#         specificity.append(spec)
#
#     f1 = [2 * safe_divide((pr * rc), (pr + rc)) for pr, rc in zip(precision, recall)]
#
#     # 更新进度条描述
#     description = (
#         f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, "
#         f"acc: {accu_num.item() / sample_num:.3f}, "
#         f"recall: {', '.join(f'{r:.3f}' for r in recall)}, "
#         f"precision: {', '.join(f'{p:.3f}' for p in precision)}, "
#         f"F1: {', '.join(f'{f:.3f}' for f in f1)}, "
#         f"specificity: {', '.join(f'{s:.3f}' for s in specificity)}, "
#         f"lr: {optimizer.param_groups[0]['lr']:.5f}"
#     )
#     data_loader.set_description(description)
#     print(description)
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num, recall, specificity, precision, f1
#
#
# @torch.no_grad()
# def evaluate(model, data_loader, device, epoch, criterion):
#     model.eval()
#     accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#
#     sample_num = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []
#
#     data_loader = tqdm(data_loader, file=sys.stdout)
#
#     for step, data in enumerate(data_loader):
#         h_images, v_images, text_data, labels, h_paths, v_paths = data
#         pred = model(h_images.to(device), v_images.to(device), text_data.to(device))
#
#         pred_classes = torch.max(pred, dim=1)[1]
#         prob = torch.nn.functional.softmax(pred, dim=1)
#
#         correct = torch.eq(pred_classes, labels.to(device))
#         accu_num += correct.sum()
#         loss = criterion(pred, labels.to(device))  # 使用传入的criterion计算损失
#         accu_loss += loss
#
#         sample_num += labels.size(0)
#         all_preds.extend(pred_classes.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#         all_probs.extend(prob.cpu().numpy())
#
#     # 计算混淆矩阵
#     num_classes = 3
#     cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
#
#     recall = [safe_divide(np.diag(cm)[i], np.sum(cm[i, :])) for i in range(num_classes)]
#     precision = [safe_divide(np.diag(cm)[i], np.sum(cm[:, i])) for i in range(num_classes)]
#
#     specificity = []
#     for i in range(num_classes):
#         TP = cm[i, i]
#         FP = np.sum(cm[:, i]) - TP
#         FN = np.sum(cm[i, :]) - TP
#         TN = np.sum(cm) - TP - FP - FN
#         spec = safe_divide(TN, TN + FP)
#         specificity.append(spec)
#
#     f1 = [2 * safe_divide((pr * rc), (pr + rc)) for pr, rc in zip(precision, recall)]
#
#     # 更新进度条描述
#     description = (
#         f"[valid epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, "
#         f"acc: {accu_num.item() / sample_num:.3f}, "
#         f"recall: {', '.join(f'{r:.3f}' for r in recall)}, "
#         f"precision: {', '.join(f'{p:.3f}' for p in precision)}, "
#         f"F1: {', '.join(f'{f:.3f}' for f in f1)}, "
#         f"specificity: {', '.join(f'{s:.3f}' for s in specificity)}, "
#     )
#
#     data_loader.set_description(description)
#     print(description)
#
#     # 计算并绘制ROC曲线
#     all_labels = label_binarize(all_labels, classes=[0, 1, 2])
#     for i in range(num_classes):
#         fpr, tpr, _ = roc_curve(all_labels[:, i], np.array(all_probs)[:, i])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, lw=2, label=f'Class {i} ROC curve (area = {roc_auc:.2f})')
#
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc='lower right')
#     plt.show()
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num, recall, specificity, precision, f1
#
# def create_lr_scheduler(optimizer,
#                         num_step: int,
#                         epochs: int,
#                         warmup=True,
#                         warmup_epochs=1,
#                         warmup_factor=1e-3,
#                         end_factor=1e-6):
#     assert num_step > 0 and epochs > 0
#     if warmup is False:
#         warmup_epochs = 0
#
#     def f(x):
#         """
#         根据step数返回一个学习率倍率因子，
#         注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
#         """
#         if warmup is True and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             # warmup过程中lr倍率因子从warmup_factor -> 1
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             current_step = (x - warmup_epochs * num_step)
#             cosine_steps = (epochs - warmup_epochs) * num_step
#             # warmup后lr倍率因子从1 -> end_factor
#             return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
#
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
#
#
# def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
#     # 记录optimize要训练的权重参数
#     parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
#                             "no_decay": {"params": [], "weight_decay": 0.}}
#
#     # 记录对应的权重名称
#     parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
#                              "no_decay": {"params": [], "weight_decay": 0.}}
#
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#
#         if len(param.shape) == 1 or name.endswith(".bias"):
#             group_name = "no_decay"
#         else:
#             group_name = "decay"
#
#         parameter_group_vars[group_name]["params"].append(param)
#         parameter_group_names[group_name]["params"].append(name)
#
#     # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
#     return list(parameter_group_vars.values())
#
#

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from _datetime import datetime
import os
import sys
import json
import pickle
import random
import math
import numpy as np
import torch
from tqdm import tqdm


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 切换后端为 TkAgg
import matplotlib.pyplot as plt


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, criterion):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    all_preds = []
    all_labels = []
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        h_images, v_images, text_data, labels, h_path, v_path = data
        pred = model(h_images.to(device), v_images.to(device), text_data.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = criterion(pred, labels.to(device))  # 使用传入的criterion计算损失
        loss.backward()
        accu_loss += loss.detach()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            return None

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        sample_num += labels.size(0)
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    num_classes = 3
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    recall = [safe_divide(np.diag(cm)[i], np.sum(cm[i, :])) for i in range(num_classes)]
    precision = [safe_divide(np.diag(cm)[i], np.sum(cm[:, i])) for i in range(num_classes)]

    specificity = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - TP - FP - FN
        spec = safe_divide(TN, TN + FP)
        specificity.append(spec)

    f1 = [2 * safe_divide((pr * rc), (pr + rc)) for pr, rc in zip(precision, recall)]

    # 更新进度条描述
    description = (
        f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, "
        f"acc: {accu_num.item() / sample_num:.3f}, "
        f"recall: {', '.join(f'{r:.3f}' for r in recall)}, "
        f"precision: {', '.join(f'{p:.3f}' for p in precision)}, "
        f"F1: {', '.join(f'{f:.3f}' for f in f1)}, "
        f"specificity: {', '.join(f'{s:.3f}' for s in specificity)}, "
        f"lr: {optimizer.param_groups[0]['lr']:.5f}"
    )
    data_loader.set_description(description)
    print(description)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, recall, specificity, precision, f1


# @torch.no_grad()
# def evaluate(model, data_loader, device, epoch, criterion):
#     model.eval()
#     accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#
#     sample_num = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []
#
#     data_loader = tqdm(data_loader, file=sys.stdout)
#
#     for step, data in enumerate(data_loader):
#         h_images, v_images, text_data, labels, h_paths, v_paths = data
#         pred = model(h_images.to(device), v_images.to(device), text_data.to(device))
#
#         pred_classes = torch.max(pred, dim=1)[1]
#         prob = torch.nn.functional.softmax(pred, dim=1)
#
#         correct = torch.eq(pred_classes, labels.to(device))
#         accu_num += correct.sum()
#         loss = criterion(pred, labels.to(device))  # 使用传入的criterion计算损失
#         accu_loss += loss
#
#         sample_num += labels.size(0)
#         all_preds.extend(pred_classes.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#         all_probs.extend(prob.cpu().numpy())
#
#     # 计算混淆矩阵
#     num_classes = 3
#     cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
#
#     recall = [safe_divide(np.diag(cm)[i], np.sum(cm[i, :])) for i in range(num_classes)]
#     precision = [safe_divide(np.diag(cm)[i], np.sum(cm[:, i])) for i in range(num_classes)]
#
#     specificity = []
#     for i in range(num_classes):
#         TP = cm[i, i]
#         FP = np.sum(cm[:, i]) - TP
#         FN = np.sum(cm[i, :]) - TP
#         TN = np.sum(cm) - TP - FP - FN
#         spec = safe_divide(TN, TN + FP)
#         specificity.append(spec)
#
#     f1 = [2 * safe_divide((pr * rc), (pr + rc)) for pr, rc in zip(precision, recall)]
#
#     # 更新进度条描述
#     description = (
#         f"[valid epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, "
#         f"acc: {accu_num.item() / sample_num:.3f}, "
#         f"recall: {', '.join(f'{r:.3f}' for r in recall)}, "
#         f"precision: {', '.join(f'{p:.3f}' for p in precision)}, "
#         f"F1: {', '.join(f'{f:.3f}' for f in f1)}, "
#         f"specificity: {', '.join(f'{s:.3f}' for s in specificity)}, "
#     )
#
#     data_loader.set_description(description)
#     print(description)
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num, recall, specificity, precision, f1
#
@torch.no_grad()
def evaluate(model, data_loader, device, epoch, criterion, log_dir='logs/'):
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    all_preds = []
    all_labels = []
    all_probs = []

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        h_images, v_images, text_data, labels, h_paths, v_paths = data
        pred = model(h_images.to(device), v_images.to(device), text_data.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        prob = torch.nn.functional.softmax(pred, dim=1)

        correct = torch.eq(pred_classes, labels.to(device))
        accu_num += correct.sum()
        loss = criterion(pred, labels.to(device))  # 使用传入的criterion计算损失
        accu_loss += loss

        sample_num += labels.size(0)
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(prob.cpu().numpy())

    # 计算混淆矩阵
    num_classes = 3
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    recall = [safe_divide(np.diag(cm)[i], np.sum(cm[i, :])) for i in range(num_classes)]
    precision = [safe_divide(np.diag(cm)[i], np.sum(cm[:, i])) for i in range(num_classes)]

    specificity = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - TP - FP - FN
        spec = safe_divide(TN, TN + FP)
        specificity.append(spec)

    f1 = [2 * safe_divide((pr * rc), (pr + rc)) for pr, rc in zip(precision, recall)]

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    # 定义标签映射
    label_map = {0: 'unsatisfactory', 1: 'stable', 2: 'good'}

    # 保存混淆矩阵
    cm_fig, cm_ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=cm_ax)
    cm_ax.set_xlabel('Predicted labels')
    cm_ax.set_ylabel('True labels')
    cm_ax.set_title('Confusion Matrix')
    # 设置x轴和y轴的标签
    cm_ax.set_xticklabels([label_map[i] for i in range(num_classes)])
    cm_ax.set_yticklabels([label_map[i] for i in range(num_classes)])
    cm_fig.savefig(os.path.join(log_dir, f'confusion_matrix_epoch_{epoch}_{timestamp}.png'))
    plt.close(cm_fig)

    # 计算并保存ROC曲线
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2])
    roc_fig, roc_ax = plt.subplots()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], np.array(all_probs)[:, i])
        roc_auc = auc(fpr, tpr)
        roc_ax.plot(fpr, tpr, lw=2, label=f'{label_map[i]} ROC curve (area = {roc_auc:.2f})')

    roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    roc_ax.set_xlim([0.0, 1.0])
    roc_ax.set_ylim([0.0, 1.05])
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('Receiver Operating Characteristic')
    roc_ax.legend(loc='lower right')
    roc_fig.savefig(os.path.join(log_dir, f'roc_curve_epoch_{epoch}_{timestamp}.png'))
    plt.close(roc_fig)

    # 更新进度条描述
    description = (
        f"[valid epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, "
        f"acc: {accu_num.item() / sample_num:.3f}, "
        f"recall: {', '.join(f'{r:.3f}' for r in recall)}, "
        f"precision: {', '.join(f'{p:.3f}' for p in precision)}, "
        f"F1: {', '.join(f'{f:.3f}' for f in f1)}, "
        f"specificity: {', '.join(f'{s:.3f}' for s in specificity)}, "
    )

    data_loader.set_description(description)
    print(description)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, recall, specificity, precision, f1

def plot_average_roc_curve(all_roc_data, num_classes):
    """
    绘制所有折的平均ROC曲线
    """
    mean_fpr = np.linspace(0, 1, 100)
    all_tprs = []
    all_aucs = []

    plt.figure(figsize=(8, 6))
    for fold, (fpr, tpr, roc_auc) in enumerate(all_roc_data, 1):
        all_aucs.append(roc_auc)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        all_tprs.append(tpr_interp)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {fold} (AUC = {roc_auc:.2f})")

    mean_tpr = np.mean(all_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=f"Mean ROC (AUC = {mean_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    绘制训练和验证损失、准确率的曲线
    """
    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def create_lr_scheduler(optimizer, num_step, epochs, warmup=True, warmup_epochs=1, warmup_factor=1e-3, end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"
        parameter_group_vars[group_name]["params"].append(param)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
