# Contrastive-learning-based-hyperspectral-image-classification
Based on contrastive learning, combined with superpixel segmentation and principal component analysis (PCA), an unsupervised feature learning framework is constructed to improve the performance of hyperspectral image classification.

# Contrastive-learning-based Hyperspectral Image Classification

基于对比学习（Contrastive Learning），结合超像素分割（Superpixel Segmentation）与主成分分析（PCA），构建了一个**无监督特征学习框架**，用于提升高光谱图像分类性能。

---

## 📁 文件介绍

由于我没有将代码模块化，整个项目比较“史山”（集中在 notebook 中），因此对每个文件做如下说明：

> **以下文件均基于 Houston2013 数据集实验**

---

### 1. `Moco.ipynb`
- 尝试使用 **MoCo（Momentum Contrast）** 实现动量式对比学习；
- 实验结果显示精度有一定提升；
- **缺点**：时间成本较高，按当前参数设置，训练时间约为原来的 **50 倍**，不是最优解。

---

### 2. `S3PCA_20dimension.ipynb`
- 切分方式：
  - 块 a + 局部 PCA 降维至 20 维（a1）
  - 块 b + 全局 PCA 降维至 20 维（b1）
- 最终 a1 和 b1 均为 20 维特征。

---

### 3. `S3PCA_40dimension.ipynb`
- 切分方式：
  - 块 a + 局部 PCA 降维 20 维
  - 块 b + 全局 PCA 降维 20 维
- 然后合并成 40 维特征。

---

### 4. `S3PCA_40dimension_OPCArandom.ipynb`
- 与 `S3PCA_40dimension.ipynb` 相同结构；
- 区别在于：**使用随机选取的样本**进行 PCA 降维（模拟更不确定的场景）。

---

### 5. `S3PCA_fulldimension.ipynb`
- **不进行降维**，保留原始维度的局部和全局特征；
- 用于对比降维与否对分类性能的影响。

---

### 6. `S3PCA_nocat.ipynb`
- 不拼接局部与全局，仅使用它们**独立训练**并评估；
- 目的是验证单一特征来源的效果。

---
=======
# **Contrastive Learning-Based Hyperspectral Image Classification**  
🚀 **A novel contrastive learning framework for hyperspectral image classification, leveraging superpixel-based local PCA and global PCA features.**  

## **📖 Overview**  
Hyperspectral image (HSI) classification is a challenging task due to the high-dimensional spectral information and limited labeled data. This project introduces a **contrastive learning-based framework** that enhances **spectral-spatial feature learning** by incorporating **superpixel segmentation and PCA transformation**.  

By constructing **positive sample pairs** from both **local (SuperPCA) and global (GlobalPCA) spectral features**, the model effectively learns **discriminative representations** for HSI classification.  

---

## **📌 Methodology**  
The proposed framework follows these key steps:  

### **1️⃣ PCA Dimensionality Reduction**  
- The raw hyperspectral data is **transformed into 40 principal components** using **Principal Component Analysis (PCA)** to reduce computational complexity while preserving essential spectral information.  

### **2️⃣ Superpixel Segmentation**  
- **Simple Linear Iterative Clustering (SLIC)** is used to segment the hyperspectral image into superpixels, capturing structural and local spatial information.  

### **3️⃣ Feature Extraction (Local & Global PCA)**  
- **SuperPCA**: PCA is applied **within each superpixel** to extract localized spectral-spatial features.  
- **GlobalPCA**: PCA is computed **across the entire training region**, generating global spectral features.  

### **4️⃣ Contrastive Learning - Cube Construction**  
To create effective contrastive learning pairs, **two feature cubes (Cube A & Cube B) are constructed**:  
- **Cube A** = **Random 20 PCA channels** + **SuperPCA (local)**  
- **Cube B** = **Remaining 20 PCA channels** + **GlobalPCA (global)**  

These cubes serve as **positive pairs**, ensuring that they represent the **same spatial region** but with **different feature perspectives**.  

### **5️⃣ Contrastive Learning Training**  
- **InfoNCE loss** is applied to **maximize the similarity between Cube A & Cube B** while **minimizing their similarity with negative samples**.  
- The model is trained in a **self-supervised manner**, reducing dependence on labeled data.  

---


