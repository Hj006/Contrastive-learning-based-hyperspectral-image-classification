#main_contrastive.py
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

from datasets_factory import get_dataset
from datasets import NeighborhoodSplitPositiveDataset
from pca_utils import apply_pca_on_candidate
from models import FeatureExtractor, ProjectionHead
from losses import contrastive_loss_ce_hard_negatives

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='whu', choices=['whu', 'paviau', 'houston2013'], help="Dataset name")
    parser.add_argument('--pca_dim', type=int, default=40, help="PCA dimension, default 40")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patch_size', type=int, default=11)
    parser.add_argument('--pth_dir', type=str, default='pth', help='Directory to save model checkpoints')
    args = parser.parse_args()

    # 1. 设置各数据集的默认路径
    DATASET_PATHS = {
        'whu': r'E:\code\-\对比学习\fx\N-SSMCL\Data\WHU-Hi-LongKou',
        'paviau': r'E:\code\-\对比学习\fx\N-SSMCL\Data',
        'houston2013': r'E:\code\-\对比学习\fx\N-SSMCL\Data\2013_DFTC'
    }
    data_path = DATASET_PATHS[args.dataset]

    # 2. 配置不同数据集参数
    if args.dataset == 'whu':
        candidate_counts = {i: 300 for i in range(1, 10)}
        dataset_kwargs = {'candidate_counts': candidate_counts, 'num_train_per_class': 15}
    elif args.dataset == 'paviau':
        candidate_counts = {1: 548, 2: 540, 3: 392, 4: 524, 5: 265, 6: 532, 7: 375, 8: 514, 9: 231}
        dataset_kwargs = {'candidate_counts': candidate_counts, 'num_train_per_class': 10}
    elif args.dataset == 'houston2013':
        dataset_kwargs = {}
    else:
        raise ValueError

    # 3. 加载数据
    hsi_data, train_truth, candidate_truth, test_truth, info = get_dataset(
        args.dataset, data_path, **dataset_kwargs)

    # 4. PCA降维
    pca_candidate_data, reduced_samples, coords, var_ratio = apply_pca_on_candidate(
        hsi_data, candidate_truth, num_components=args.pca_dim)

    pca_data_tensor = torch.tensor(pca_candidate_data).float()
    dataset = NeighborhoodSplitPositiveDataset(pca_data_tensor, coords, patch_size=args.patch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    feature_extractor = FeatureExtractor(input_channels=args.pca_dim).cuda()
    projection_head = ProjectionHead(input_dim=128, output_dim=8).cuda()
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(projection_head.parameters()), lr=1e-4)

    os.makedirs(args.pth_dir, exist_ok=True)
    loss_values = []
    for epoch in range(args.num_epochs):
        feature_extractor.train()
        projection_head.train()
        epoch_loss = 0
        for cube_a, cube_b in dataloader:
            cube_a, cube_b = cube_a.cuda(), cube_b.cuda()
            features_a = feature_extractor(cube_a)
            features_b = feature_extractor(cube_b)
            proj_a = projection_head(features_a)
            proj_b = projection_head(features_b)
            loss = contrastive_loss_ce_hard_negatives(proj_a, proj_b, temperature=1, num_negatives=5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")
        # 保存模型
        if (epoch + 1) in [50, 100, 150]:
            model_path = f"{args.pth_dir}/{args.dataset}_lin_{epoch+1}_{args.pca_dim}e.pth"
            torch.save({
                'epoch': epoch + 1,
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_path)
            print(f"Model saved at epoch {epoch+1} to {model_path}")

    # 5. 保存损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.num_epochs + 1), loss_values, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f'{args.pth_dir}/{args.dataset}_lin_{args.num_epochs}_{args.pca_dim}e.png')
    plt.show()

if __name__ == "__main__":
    main()
'''
# 用WHU
python main_contrastive.py --dataset whu

# 用PaviaU
python main_contrastive.py --dataset paviau

# 用Houston2013
python main_contrastive.py --dataset houston2013

# 如要pca_dim=60则加参数
python main_contrastive.py --dataset whu --pca_dim 60

# 指定模型目录
python main_contrastive.py --dataset whu --pth_dir my_pth



'''