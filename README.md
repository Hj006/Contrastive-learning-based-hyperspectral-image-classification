# Contrastive Learning-Based Hyperspectral Image Classification

### 项目简介

本项目提出了一种面向高光谱图像分类任务的无监督特征学习框架：**基于邻域的光谱-空间混合对比学习（N-SSMCL）**。该方法结合了主成分分析（PCA）、空间邻域结构、通道切分与交换机制，创新性地设计了更具判别性的正样本对构造策略。

具体而言，模型首先通过 PCA 对高光谱数据进行降维以压缩冗余信息。然后在局部空间区域（Patch）中，引入邻域像元（上下左右）作为对比学习的配对基础，并对其光谱通道进行切分与交叉交换，生成融合光谱差异性与空间一致性的**混合 Patch 对**作为正样本对。此策略增强了特征的判别性，同时缓解了邻域异类样本可能带来的标签噪声影响。

此外，在对比损失函数中，提出了**动态简易负样本挖掘机制（InfoNCE-Easy）**，自动选择与锚点样本最不相似的负样本，提高对比学习的优化效率与鲁棒性。整套方法在预训练阶段无需标签，支持少量标注样本的下游分类微调，展现出优良的泛化能力和实际应用潜力。

---

### Project Introduction

This project proposes an unsupervised learning framework named **Neighbor-based Spectral-Spatial Mixed Contrastive Learning (N-SSMCL)**, designed for hyperspectral image classification. The method integrates **Principal Component Analysis (PCA)**, **spatial neighborhood learning**, and a **channel-split-and-exchange strategy** to generate discriminative positive sample pairs for contrastive learning.

Specifically, PCA is first applied to reduce spectral dimensionality. Then, centered patches and their four neighboring patches (up, down, left, right) are extracted from the image. These patches undergo spectral channel splitting and swapping operations to create **mixed patches** that combine spectral diversity and spatial consistency. These mixed patch pairs serve as **positive pairs**, enabling the model to learn more robust and discriminative representations.

A novel **InfoNCE-Easy loss** is also introduced, which dynamically selects the hardest *easy* negative samples—those least similar to the anchor—during each training batch. This improves contrastive optimization efficiency and reduces the risk of overfitting on outlier negatives.

The encoder architecture is based on convolutional neural networks, with a projection head mapping high-dimensional features into the contrastive space. After self-supervised pretraining using only unlabeled data, the model can be fine-tuned on downstream classification tasks with minimal labeled samples, demonstrating strong generalization and practicality.

---

项目包含两个部分：

- **Reproduce-the-original-paper**：对原论文方法的实验复现，包括探索性 notebook 实验。
- **N-SSMCL**：将本文提出的方法进行模块化实现，支持多个数据集训练和评估，包含主干模型、对比损失、分类训练与多种 Baseline 实验。

The project consists of two parts:

- **Reproduce-the-original-paper**: Experimental reproduction of the original paper method, including exploratory notebook experiments.
- **N-SSMCL**: Modularize the method proposed in this paper, support training and evaluation of multiple datasets, including backbone model, contrast loss, classification training and multiple baseline experiments.

---

## 项目结构 Project Structure

```text
.
.
├── Reproduce-the-original-paper/         # 原论文实验复现（集中在 Notebook 中）
│   ├── Houston2013/                      # 数据目录
│   ├── pth/                              # 模型权重输出（复现部分）
│   ├── Moco.ipynb                        # 动量对比学习实验
│   ├── roll_50_S3PCA_40dimension.ipynb   # 基于滚动窗口的特征提取实验
│   ├── S3PCA_20dimension.ipynb           # 局部/全局 20维降维特征对比
│   ├── S3PCA_40dimension.ipynb           # 合并生成40维的特征结构实验
│   ├── S3PCA_40dimension_OPCArandom.ipynb # 随机选样降维实验
│   ├── S3PCA_fulldimension.ipynb         # 不降维的特征实验
│   ├── S3PCA_nocat.ipynb                 # 不拼接特征的对比实验
│   ├── 不标准.ipynb                       # 命名不标准的实验记录
│   ├── 复现.ipynb                         # 原文方法复现记录
│   └── 集成训练.ipynb                     # 集成模型训练尝试
│
├── N-SSMCL/                              # 本文提出方法的模块化实现
│   ├── main_contrastive.py              # 对比学习预训练主程序
│   ├── main_classification.py           # 下游分类任务主程序
│   ├── datasets.py                      # 数据集读取与构建逻辑
│   ├── datasets_factory.py              # 数据集工厂，统一接口
│   ├── models.py                        # 主干网络、投影头、分类头定义
│   ├── pca_utils.py                     # PCA 工具模块
│   ├── losses.py                        # 对比损失函数定义
│   ├── train_eval.py                    # 微调训练与评估脚本
│   ├── pth/                             # 训练生成的模型权重与日志
│   ├── Data/                            # 数据集目录（需手动准备）
│   ├── Othermethods/                    # Baseline 的 Jupyter 实现（Baselines）
│   └── README.md                        # 模块化代码说明文档
│
└── README.md                            # 项目总览（你正在阅读的主文档）

````

---

## 本项目解决了什么问题 What This Project Does

当然可以。你这段内容描述得已经不错了，但略显简化，无法准确覆盖你论文中提出的**邻域引导、通道切分交换、动态负样本挖掘等关键创新点**。以下是我为你精炼与增强后的版本，**更准确表达你提出的内容**，同时保留中英文对照风格：

---

## 本项目解决了什么问题 What This Project Does

### 中文说明：

针对遥感高光谱图像在分类任务中面临的**高维冗余特征**和**标注样本稀缺**问题，本项目提出了一种**基于邻域的光谱-空间混合对比学习框架（N-SSMCL）**，具体包括以下关键设计：

 **PCA降维**：对原始高光谱数据进行主成分分析，有效压缩维度、保留主信息；

**邻域引导 Patch 构造**：以目标像素为中心，提取上下左右邻域 Patch，构造空间局部关联；

**通道切分与交换机制**：对 Patch 进行光谱维度切分与互换，生成具有**光谱多样性与空间一致性**的混合正样本对；

**动态简易负样本挖掘（InfoNCE-Easy）**：优化负样本选择策略，提升特征判别力与训练稳定性；

**模块化实现 + 可复现 Baseline**：支持多数据集实验、完整复现过程及对比方法，便于研究者扩展与验证。

---

### 📖 English Summary:

To address the challenges of **high spectral dimensionality** and **limited labeled samples** in hyperspectral image classification, this project proposes a novel framework: **Neighbor-based Spectral-Spatial Mixed Contrastive Learning (N-SSMCL)**, featuring the following innovations:

**PCA for dimensionality reduction**, preserving essential spectral information while reducing computational cost;

**Neighborhood-guided patch construction**, extracting four spatial neighbors (up/down/left/right) around each pixel to capture local spatial context;

**Channel-split and exchange**, generating positive pairs by exchanging spectral channels between central and neighbor patches to enhance diversity and spatial consistency;

**Dynamic Easy Negative Mining (InfoNCE-Easy)**, selecting the least similar negatives to each anchor during contrastive training for better optimization and robustness;

**Modular implementation and reproducible baselines**, supporting multiple datasets, plug-and-play training, and easy extension for further research.


---

## 支持的数据集 Supported Datasets

| Dataset Name     | Description                                   | Download Link                                                                        |
| ---------------- | --------------------------------------------- | ------------------------------------------------------------------------------------ |
| WHU-Hi-LongKou   | Small-area high-resolution hyperspectral data | [WHU-Hi 官网](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)                    |
| Pavia University | Urban scene captured by ROSIS sensor          | [PaviaU - PapersWithCode](https://paperswithcode.com/dataset/pavia-university)       |
| Houston 2013     | IEEE GRSS Contest Dataset                     | [Houston2013](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/) |

> 请将数据集解压后放置于 `N-SSMCL/Data/` 目录中，并确保文件名与代码中的 `DATASET_PATHS` 匹配。

---

## 环境依赖 Environment

建议使用 conda 虚拟环境：

```bash
conda create -n nssmcl python=3.9.16
conda activate nssmcl
pip install torch numpy scikit-learn matplotlib scipy scikit-image rasterio
```

---

## 快速开始 Quick Start

### 1. 预训练对比学习模型 Contrastive Pretraining

```bash
cd N-SSMCL
python main_contrastive.py --dataset whu
```

### 2. 微调分类模型 Fine-tuning for Classification

```bash
python main_classification.py --dataset whu --ckpt_name model_whu.pth
```

所有参数均支持自定义，例如：

```bash
--pca_dim 40 --batch_size 64 --num_epochs 100 --pth_dir ./pth
```

---

## Baseline 对比方法 Other Methods

`N-SSMCL/Othermethods/` 中包含以下 Jupyter 实验：

* `CS`：通道切分构造正样本；
* `CS-DR`：结合 PCA / LDA 维度压缩；
* `NB`：像素邻域方法；
* `PCA_LDA`：常规降维对比。

这些方法均基于公开数据集，可直接运行评估。

---

## 输出结果 Outputs

* 所有训练模型保存在 `pth/` 目录；
* 输出包括：

  * Loss & Accuracy 曲线；
  * OA, AA, Kappa 系数；
  * 每类精度等评估指标。

---

## 引用与致谢 Citation & Acknowledgements

感谢以下开源资源与数据支持：

* PyTorch, scikit-learn, rasterio 等优秀框架；
* 公共数据集发布者（WHU-Hi, PaviaU, Houston2013）；
* 所有高光谱遥感、对比学习领域相关研究。

如果本项目对你的研究有帮助，请在你的论文或报告中引用或致谢。

---

##  Citation & Acknowledgements

I sincerely acknowledge the contributions of the following open-source resources and datasets:

* Open-source frameworks such as **PyTorch**, **scikit-learn**, **rasterio**, and others;
* Public hyperspectral datasets including **WHU-Hi-LongKou**, **Pavia University**, and **Houston2013**;
* Prior research works in the fields of **hyperspectral remote sensing**, **contrastive learning**, and **unsupervised representation learning**.

If you find this project helpful for your research or applications, please consider citing or acknowledging it in your publications.

---

## 📬 联系方式 Contact

* Email: [jiangxiaobai1142@gmail.com](mailto:jiangxiaobai1142@gmail.com)

欢迎反馈 bug、交流合作或提出改进建议！

Welcome your feedback, suggestions for improvement, or feedback on bugs!

---
