# main_classification.py
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse

from datasets_factory import get_dataset
from datasets import ClassificationDataset
from pca_utils import apply_pca_train_only
from models import FeatureExtractor, ClassificationHead
from train_eval import train_classification_model, evaluate_classification_model_with_details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='whu', choices=['whu', 'paviau', 'houston2013'], help="Dataset name")
    parser.add_argument('--pca_dim', type=int, default=40, help="PCA dimension, default 40")
    parser.add_argument('--patch_size', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--pth_dir', type=str, default='pth', help='Directory for model weights')
    parser.add_argument('--ckpt_name', type=str, default=None, help="Checkpoint filename. If None, use dataset defaults")
    args = parser.parse_args()

    # 1. 数据集默认路径
    DATASET_PATHS = {
        'whu': r'E:\code\-\对比学习\fx\N-SSMCL\Data\WHU-Hi-LongKou',
        'paviau': r'E:\code\-\对比学习\fx\N-SSMCL\Data',
        'houston2013': r'E:\code\-\对比学习\fx\N-SSMCL\Data\2013_DFTC'
    }
    data_path = DATASET_PATHS[args.dataset]

    # 2. 数据集参数
    if args.dataset == 'whu':
        candidate_counts = {i: 300 for i in range(1, 10)}
        dataset_kwargs = {'candidate_counts': candidate_counts, 'num_train_per_class': 15}
        num_classes = 9
    elif args.dataset == 'paviau':
        candidate_counts = {1: 548, 2: 540, 3: 392, 4: 524, 5: 265, 6: 532, 7: 375, 8: 514, 9: 231}
        dataset_kwargs = {'candidate_counts': candidate_counts, 'num_train_per_class': 10}
        num_classes = 9
    elif args.dataset == 'houston2013':
        dataset_kwargs = {}
        num_classes = 15
    else:
        raise ValueError

    # 3. 数据集加载
    hsi_data, train_truth, candidate_truth, test_truth, info = get_dataset(
        args.dataset, data_path, **dataset_kwargs
    )

    # 4. 标签提取
    train_labels = [(r, c, label - 1) for r, c, label in zip(train_truth.row, train_truth.col, train_truth.data)]
    test_labels = [(r, c, label - 1) for r, c, label in zip(test_truth.row, test_truth.col, test_truth.data)]

    # 5. PCA降维
    pca_data, explained_variance_ratio = apply_pca_train_only(hsi_data, train_truth, num_components=args.pca_dim)

    train_dataset = ClassificationDataset(pca_data, train_labels, patch_size=args.patch_size)
    test_dataset = ClassificationDataset(pca_data, test_labels, patch_size=args.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. 加载对比学习主干特征
    if args.ckpt_name is not None:
        checkpoint_path = os.path.join(args.pth_dir, args.ckpt_name)
    else:
        # 默认命名规则
        checkpoint_path = os.path.join(args.pth_dir, f"{args.dataset}_lin_50_{args.pca_dim}e.pth")

    print(f"Loading backbone weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    feature_extractor = FeatureExtractor(input_channels=args.pca_dim).cuda()
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    for param in feature_extractor.parameters():
        param.requires_grad = True

    classification_head = ClassificationHead(input_dim=128, num_classes=num_classes).cuda()
    optimizer = torch.optim.Adam([
        {"params": feature_extractor.parameters(), "lr": 1e-4},
        {"params": classification_head.parameters(), "lr": 1e-3},
    ])
    criterion = torch.nn.CrossEntropyLoss()

    # 7. 微调+测试
    train_classification_model(
        feature_extractor, classification_head, train_loader, optimizer, criterion, num_epochs=args.num_epochs
    )
    evaluate_classification_model_with_details(
        feature_extractor, classification_head, test_loader, num_classes=num_classes
    )

if __name__ == "__main__":
    main()

'''
python main_classification.py --dataset whu
python main_classification.py --dataset paviau
python main_classification.py --dataset whu --ckpt_name mywhu_60e.pth

python main_classification.py --dataset whu --pth_dir model_weights --ckpt_name customwhu.pth

'''