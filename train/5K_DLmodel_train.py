import datetime
import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torchvision.models import ResNet50_Weights

from dataset.DL_dataset import DLDataSet
from loss import CombinedLoss
from model.cross_token_attention import MMF
from utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from model.DLmodel import MultimodalFusionNet


def main(args):
    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    if not os.path.exists("../weights"):
        os.makedirs("../weights")

    data_dir = 'E:\\预测数据\\data'
    dataset = DLDataSet(data_dir)

    # 使用5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    total_best_val_loss = 0.0
    total_best_val_acc = 0.0
    total_best_val_recall = []
    total_best_val_specificity = []
    total_best_val_precision = []
    total_best_val_f1 = []

    fold = 1
    results = []

    for train_index, val_index in kf.split(dataset):
        # if fold < 4:  # 跳过前三次
        #     fold += 1]
        #     continue  # 直接跳到下一次迭代

        print(f"Fold: {fold}")

        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])  # number of workers
        print(f'Using {nw} dataloader workers per process.')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,


                                                 num_workers=nw)

        # 获取文本特征的大小
        text_feature_size = dataset.processed_features.shape[1]   #processed_features [376, 27]    text_feature_size 27
        num_classes = len(dataset.classes)
        model = MultimodalFusionNet(num_classes, text_feature_size)
        model = model.to(device)
        # print("model", model)

        pg = get_params_groups(model, weight_decay=args.wd)
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                           warmup=True, warmup_epochs=1)


        # 初始化每折的最佳验证结果
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_recall = []
        best_val_specificity = []
        best_val_precision = []
        best_val_f1 = []

        # 计算每个类别的权重
        weights = [6, 3, 1]
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        best_metrics = {"val_loss": float('inf'), "val_acc": 0, "val_recall": [], "val_specificity": [],
                        "val_precision": [], "val_f1": []}

        # criterion = CombinedLoss(alpha=1.0, beta=1.0, gamma=2.0, weight=None, reduction='mean')
        criterion = nn.CrossEntropyLoss(weight=None, reduction='mean', label_smoothing=0.1)

        for epoch in range(args.epochs):
            # train
            train_loss, train_acc, train_recall, train_specificity, train_precision, train_f1 = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                criterion=criterion)

            # validate
            val_loss, val_acc, val_recall, val_specificity, val_precision, val_f1 = evaluate(model=model,
                                                                                             data_loader=val_loader,
                                                                                             device=device,
                                                                                             epoch=epoch,
                                                                                             criterion=criterion)
            model_name = os.path.basename(opt.weights).split('.')[0] if opt.weights else "resnet"
            if best_metrics["val_acc"] < val_acc:
                print("保存模型", "第", epoch, "个epoch")
                current_time = datetime.datetime.now().strftime("%Y-%m-%d")
                model_filename = f"5K_fold_best_{model_name}_{fold}_fold_{current_time}.pth"
                model_path = f"../weights/{model_filename}"
                # torch.save(model.state_dict(), f"../weights/5K_fold_best_Mymodel_model_{fold}_fold.pth")
                # torch.save(model.state_dict(), model_path)
                best_metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_recall": val_recall,
                                "val_specificity": val_specificity, "val_precision": val_precision, "val_f1": val_f1,
                                "epoch": epoch}

                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_recall = val_recall
                best_val_specificity = val_specificity
                best_val_precision = val_precision
                best_val_f1 = val_f1

        results.append(best_metrics)

        total_best_val_loss += best_val_loss
        total_best_val_acc += best_val_acc
        total_best_val_recall.append(best_val_recall)
        total_best_val_specificity.append(best_val_specificity)
        total_best_val_precision.append(best_val_precision)
        total_best_val_f1.append(best_val_f1)

        fold += 1

    def compute_average(metrics_list):
        sum_metrics = [sum(metrics) for metrics in zip(*metrics_list)]
        return [m / len(metrics_list) for m in sum_metrics]

    avg_best_val_loss = total_best_val_loss / 5
    avg_best_val_acc = total_best_val_acc / 5
    avg_best_val_recall = compute_average(total_best_val_recall)
    avg_best_val_specificity = compute_average(total_best_val_specificity)
    avg_best_val_precision = compute_average(total_best_val_precision)
    avg_best_val_f1 = compute_average(total_best_val_f1)


    print(f"Average Best Validation Loss: {avg_best_val_loss}")
    print(f"Average Best Validation Accuracy: {avg_best_val_acc}")
    print(f"Average Best Validation Recall: {avg_best_val_recall}, {sum(avg_best_val_recall) / 3}")
    print(f"Average Best Validation Specificity: {avg_best_val_specificity}, {sum(avg_best_val_specificity) / 3}")
    print(f"Average Best Validation Precision: {avg_best_val_precision}, {sum(avg_best_val_precision) / 3}")
    print(f"Average Best Validation F1 Score: {avg_best_val_f1}, {sum(avg_best_val_f1) / 3}")


    for idx, res in enumerate(results, 1):
        print(
            f"Fold {idx}: Loss={res['val_loss']} Accuracy={res['val_acc']} recall={res['val_recall']}, "
            f"specificity={res['val_specificity']} "
            f"val_precision={res['val_precision']} epoch={res['epoch']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--resnet_layers', type=int, default=50)
    parser.add_argument('--output_dim', type=int, default=256)
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    # # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
