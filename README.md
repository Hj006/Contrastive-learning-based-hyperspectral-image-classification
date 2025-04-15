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


