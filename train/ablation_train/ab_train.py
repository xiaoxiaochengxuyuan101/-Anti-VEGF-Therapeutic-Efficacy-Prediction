import os
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
from dataset.dataset import MyDataSet

from utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate


# 实验1：移除跨模态Transformer
from model.ablation.CTT_NoTransformer import CTT_NoTransformer
# 实验2：移除文本特征编码器
from model.ablation.CTT_NoText import CTT_NoText
# 实验3：单一模态
from model.ablation.CTT_OnlyHor import CTT_OnlyHor
# 实验4：仅使用transformer，不交互
from model.ablation.CTT_NoCrossInteraction import CTT_NoInteraction


def main(args, experiment):
    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for training.")

    # 加载数据集
    data_dir = 'E:\\预测数据\\data'
    dataset = MyDataSet(data_dir)

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    results = []  # 保存每折最佳结果
    final_metrics_list = []  # 保存每折最后一轮结果

    for train_index, val_index in kf.split(dataset):
        print(f"Fold {fold}")

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

        # 根据实验选择模型
        if experiment == "no_transformer":
            print("Using CTT_NoTransformer")
            model = CTT_NoTransformer(args)
        elif experiment == "no_text":
            print("Using CTT_NoText")
            model = CTT_NoText(args)
        elif experiment == "only_hor":
            print("Using CTT_OnlyHor")
            model = CTT_OnlyHor(args)
        elif experiment == "no_cross":
            print("Using CTT_NoCrossInteraction")
            model = CTT_NoInteraction(args)
        else:
            raise ValueError("Invalid experiment type")

        model = model.to(device)

        # 优化器与学习率调度器
        pg = get_params_groups(model, weight_decay=args.wd)
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1)

        # 损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 初始化保存最佳和最后一轮结果的变量
        best_metrics = {"val_loss": float('inf'), "val_acc": 0, "val_recall": [], "val_specificity": [],
                        "val_precision": [], "val_f1": [], "epoch": 0}
        final_metrics = {"val_loss": 0.0, "val_acc": 0.0, "val_recall": [], "val_specificity": [],
                         "val_precision": [], "val_f1": []}

        for epoch in range(args.epochs):
            # Train
            train_loss, train_acc, train_recall, train_specificity, train_precision, train_f1 = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                criterion=criterion)

            # Validate
            val_loss, val_acc, val_recall, val_specificity, val_precision, val_f1 = evaluate(model=model,
                                                                                             data_loader=val_loader,
                                                                                             device=device,
                                                                                             epoch=epoch,
                                                                                             criterion=criterion)

            # 保存最后一轮的验证结果
            if epoch == args.epochs - 1:
                final_metrics["val_loss"] = val_loss
                final_metrics["val_acc"] = val_acc
                final_metrics["val_recall"] = val_recall
                final_metrics["val_specificity"] = val_specificity
                final_metrics["val_precision"] = val_precision
                final_metrics["val_f1"] = val_f1

            # 更新最佳结果
            if best_metrics["val_acc"] < val_acc:
                print("保存模型", "第", epoch, "个epoch")
                best_metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_recall": val_recall,
                                "val_specificity": val_specificity, "val_precision": val_precision, "val_f1": val_f1,
                                "epoch": epoch}

        # 保存每折的最佳结果和最后一轮结果
        results.append(best_metrics)
        final_metrics_list.append(final_metrics)

        fold += 1

    # 计算平均结果的函数
    def compute_average(metrics_list):
        num_folds = len(metrics_list)
        avg_metrics = {
            "val_loss": sum([x["val_loss"] for x in metrics_list]) / num_folds,
            "val_acc": sum([x["val_acc"] for x in metrics_list]) / num_folds,
            "val_recall": [sum(x["val_recall"][i] for x in metrics_list) / num_folds for i in range(len(metrics_list[0]["val_recall"]))],
            "val_specificity": [sum(x["val_specificity"][i] for x in metrics_list) / num_folds for i in range(len(metrics_list[0]["val_specificity"]))],
            "val_precision": [sum(x["val_precision"][i] for x in metrics_list) / num_folds for i in range(len(metrics_list[0]["val_precision"]))],
            "val_f1": [sum(x["val_f1"][i] for x in metrics_list) / num_folds for i in range(len(metrics_list[0]["val_f1"]))]
        }
        return avg_metrics

    # 计算平均最佳结果
    avg_best_metrics = compute_average(results)
    # 计算平均最后一轮结果
    avg_final_metrics = compute_average(final_metrics_list)

    # 打印最佳结果
    print("\nAverage Best Results Across 5 Folds:")
    print(f"Average Best Validation Loss: {avg_best_metrics['val_loss']:.4f}")
    print(f"Average Best Validation Accuracy: {avg_best_metrics['val_acc']:.4f}")
    print(f"Average Best Validation Recall: {avg_best_metrics['val_recall']}, {sum(avg_best_metrics['val_recall']) / len(avg_best_metrics['val_recall']):.4f}")
    print(f"Average Best Validation Specificity: {avg_best_metrics['val_specificity']}, {sum(avg_best_metrics['val_specificity']) / len(avg_best_metrics['val_specificity']):.4f}")
    print(f"Average Best Validation Precision: {avg_best_metrics['val_precision']}, {sum(avg_best_metrics['val_precision']) / len(avg_best_metrics['val_precision']):.4f}")
    print(f"Average Best Validation F1 Score: {avg_best_metrics['val_f1']}, {sum(avg_best_metrics['val_f1']) / len(avg_best_metrics['val_f1']):.4f}")

    for idx, res in enumerate(results, 1):
        print(
            f"Fold {idx}: Loss={res['val_loss']} Accuracy={res['val_acc']} recall={res['val_recall']}, "
            f"specificity={res['val_specificity']} "
            f"val_precision={res['val_precision']} epoch={res['epoch']}")


    # 打印最后一轮结果
    print("\nFinal Average Results Across 5 Folds:")
    print(f"Average Final Validation Loss: {avg_final_metrics['val_loss']:.4f}")
    print(f"Average Final Validation Accuracy: {avg_final_metrics['val_acc']:.4f}")
    print(f"Average Final Validation Recall: {avg_final_metrics['val_recall']}, {sum(avg_final_metrics['val_recall']) / len(avg_final_metrics['val_recall']):.4f}")
    print(f"Average Final Validation Specificity: {avg_final_metrics['val_specificity']}, {sum(avg_final_metrics['val_specificity']) / len(avg_final_metrics['val_specificity']):.4f}")
    print(f"Average Final Validation Precision: {avg_final_metrics['val_precision']}, {sum(avg_final_metrics['val_precision']) / len(avg_final_metrics['val_precision']):.4f}")
    print(f"Average Final Validation F1 Score: {avg_final_metrics['val_f1']}, {sum(avg_final_metrics['val_f1']) / len(avg_final_metrics['val_f1']):.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--convnext_layers', type=str, default='tiny')
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--device', default='cuda:0', help='Device id (i.e. 0 or cpu)')

    opt = parser.parse_args()

    # 选择要运行的实验：
    # "no_transformer" 移除Transformer, "no_text" 移除文本, "only_hor" 仅水平图像, "only_text" 仅文本、"no_cross" 仅文本
    experiment_type = "no_text"  # 修改此处选择不同实验
    main(opt, experiment_type)
