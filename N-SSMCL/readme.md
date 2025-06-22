

# N-SSMCL: Hyperspectral Image Contrastive Learning and Classification

本项目实现了高光谱遥感影像的对比学习和分类，包括主干特征提取、投影头、分类头等模块，支持WHU-Hi-LongKou、PaviaU和Houston2013等经典数据集。
This project implements contrastive learning and supervised classification for hyperspectral remote sensing images, with support for WHU-Hi-LongKou, PaviaU, and Houston2013 datasets.

---

## 目录 Contents

* [环境依赖 Environment](#环境依赖-environment)
* [文件结构 File Structure](#文件结构-file-structure)
* [使用方法 Usage](#使用方法-usage)

  * [1. 对比学习特征预训练 Contrastive Pretraining](#1-对比学习特征预训练-contrastive-pretraining)
  * [2. 微调与分类 Fine-tuning & Classification](#2-微调与分类-fine-tuning--classification)
* [自定义参数 Customization](#自定义参数-customization)
* [结果与日志 Results](#结果与日志-results)
* [致谢 Acknowledgements](#致谢-acknowledgements)

---

## 环境依赖 Environment

* Python 3.7–3.10（推荐 3.8 或 3.9，本项目开发环境为 Python 3.9.16）
* PyTorch 1.7+
* numpy
* scikit-learn
* matplotlib
* scipy
* scikit-image
* rasterio

建议使用Anaconda创建虚拟环境。 It is recommended to use Anaconda to create a virtual environment:

```bash
conda create -n nssmcl python=3.9.16
conda activate nssmcl
pip install torch numpy scikit-learn matplotlib scipy scikit-image rasterio
```

---

## 文件结构 File Structure

```text
.
├── main_contrastive.py         # 对比学习预训练主程序
├── main_classification.py      # 下游分类任务主程序
├── datasets.py                 # 数据集读取与构建
├── datasets_factory.py         # 数据集工厂，统一接口
├── models.py                   # 主干特征提取、投影头、分类头
├── pca_utils.py                # PCA降维工具
├── losses.py                   # 对比损失函数
├── train_eval.py               # 分类训练和评估
├── pth/                        # 模型权重及训练曲线（自动生成）
├── Data/                       # 数据文件夹（请参考下文数据准备）
└── Othermethods/               # 传统/对比实验方法的Jupyter代码 (Baselines and classic methods in Jupyter notebooks)
```


---

## 其他方法与对比实验 Other Methods and Baseline Experiments

本项目的 `Othermethods/` 文件夹中包含了各类传统或对比方法在不同数据集上的 Jupyter Notebook 实现，便于与本项目主方法进行结果对比和复现。各方法含义如下：

* **CS**（Channel Split）：通道切分构造正样本对，直接分割光谱通道。
* **CS-DR**（Channel Split with Dimensionality Reduction）：通道切分结合降维（如PCA/LDA）后构造正样本对。
* **NB**（Neighborhood-Based）：利用像素空间邻域信息构造正样本对的传统方法。
* **PCA\_LDA**：分别用PCA或LDA降维后构造正样本对。

The `Othermethods/` folder contains Jupyter Notebooks implementing baseline and classical methods on different datasets, facilitating reproduction and comparison with the main approach. The meaning of each method is as follows:

* **CS** (Channel Split): Constructing positive pairs by splitting spectral channels directly.
* **CS-DR** (Channel Split with Dimensionality Reduction): Channel splitting combined with dimensionality reduction (such as PCA or LDA) for positive pair construction.
* **NB** (Neighborhood-Based): Traditional method constructing positive pairs using spatial neighborhood information.
* **PCA\_LDA**: Constructing positive pairs after applying PCA or LDA dimensionality reduction.


---

## 使用方法 Usage

### 1. 对比学习特征预训练 Contrastive Pretraining

对比学习用于预训练主干特征提取器，保存模型权重。
Contrastive learning is used to pretrain the backbone feature extractor and save the model weights.

**示例 Examples：**

```bash
python main_contrastive.py --dataset whu
python main_contrastive.py --dataset paviau
python main_contrastive.py --dataset houston2013
```

* 支持参数如 `--pca_dim` (PCA降维维数), `--batch_size`, `--num_epochs`, `--pth_dir` (模型保存目录)

* 默认权重保存在 `pth/` 目录下

* Supports parameters such as `--pca_dim` (PCA output dimension), `--batch_size`, `--num_epochs`, `--pth_dir` (directory to save models)

* Model weights are saved by default in the `pth/` directory

---

### 2. 微调与分类 Fine-tuning & Classification

在有标签的小样本下游任务上微调特征提取器并训练分类头。
Fine-tune the backbone and train the classification head on downstream few-shot classification tasks.

**示例 Examples：**

```bash
python main_classification.py --dataset whu
python main_classification.py --dataset paviau
python main_classification.py --dataset houston2013
```

* 支持参数如 `--pca_dim`, `--num_epochs`, `--batch_size`, `--ckpt_name` (指定预训练权重名), `--pth_dir`

* 输出Overall/Per-class accuracy, Kappa系数等指标

* Supports parameters such as `--pca_dim`, `--num_epochs`, `--batch_size`, `--ckpt_name` (specify pre-trained weights), `--pth_dir`

* Outputs Overall Accuracy, Per-class Accuracy, Kappa coefficient and other metrics

---

## 自定义参数 Customization

常用参数如下，所有脚本均可加`--help`查看：
The commonly used parameters are listed below. You can add `--help` to any script for details.

|       参数      |                  说明 (Description)                 | 默认值 (Default) |
| :-----------: | :-----------------------------------------------: | :-----------: |
|   --dataset   |      数据集名称 (Dataset: whu/paviau/houston2013)      |      whu      |
|   --pca\_dim  |           PCA降维维数 (PCA output dimension)          |       40      |
| --batch\_size |                  批大小 (Batch size)                 |    64 / 128   |
| --num\_epochs |          训练轮数 (Number of training epochs)         |    50 / 200   |
| --patch\_size |                Patch尺寸 (Patch size)               |       11      |
|   --pth\_dir  |           权重保存目录 (Model save directory)           |      pth      |
|  --ckpt\_name | 预训练权重文件名 (Pre-trained weights for classification) |   自动推断(Auto)  |

---

## 结果与日志 Results

* 训练过程中会输出 loss、accuracy 等日志。
  Training logs such as loss and accuracy are printed during training.
* 所有模型参数和损失曲线将保存在 `pth/` 目录。
  All model weights and loss curves are saved in the `pth/` directory.
* 分类任务会输出 Overall Accuracy、Average Accuracy、Kappa系数以及每类精度。
  The classification task outputs Overall Accuracy, Average Accuracy, Kappa coefficient, and per-class accuracy.

---


## 数据准备 Data Preparation

将各数据集（如 WHU-Hi-LongKou.tif 及其 gt 文件，PaviaU.mat 及其 gt 文件，Houston2013 官方数据等）放在项目的 `Data/` 文件夹下。
请确保各文件夹和文件名与脚本中的 `DATASET_PATHS` 字典配置一致。

Place all required datasets (such as WHU-Hi-LongKou.tif and its ground truth file, PaviaU.mat and its ground truth file, Houston2013 official data, etc.) into the `Data/` directory of the project.
Please make sure that the folder structure and filenames match the configuration in the `DATASET_PATHS` dictionary inside the code.


---

## 致谢 Acknowledgements

本项目得益于高光谱遥感影像处理、对比学习及小样本分类等领域的众多开源文献与资源。在此，感谢所有相关研究工作的贡献者，以及PyTorch、scikit-learn、rasterio等优秀开源社区。
特别感谢各公开数据集（WHU-Hi-LongKou, PaviaU, Houston2013）的发布者为研究提供支持。

如本项目对您的研究或工作有所帮助，欢迎在您的论文或成果中进行引用和致谢。

This project is inspired by a wide range of open-source research in hyperspectral remote sensing, contrastive learning, and few-shot classification. We gratefully acknowledge the contributors of related works, as well as the developers of PyTorch, scikit-learn, rasterio, and other open-source libraries.
We also thank the providers of public datasets (WHU-Hi-LongKou, PaviaU, Houston2013) for making this research possible.

If you find this project helpful for your research or work, please consider citing or acknowledging it in your publications.

---

如需进一步定制、报告bug或交流合作，请联系：

* **Email**: [jiangxiaobai1142@gmail.com](mailto:jiangxiaobai1142@gmail.com)

For questions, collaboration, or bug reports, please contact:

* **Email**: [jiangxiaobai1142@gmail.com](mailto:jiangxiaobai1142@gmail.com)

