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

## **📂 Project Structure**  
```
📁 Contrastive-Learning-HSI
│── 📁 data/                  # Hyperspectral dataset (e.g., Houston2013)
│── 📁 models/                # Feature extractor & projection head
│── 📁 utils/                 # Data processing functions (PCA, superpixels, cube extraction)
│── 📁 results/               # Training logs & visualizations
│── train.py                  # Training script for contrastive learning
│── infer.py                  # Inference script for testing the trained model
│── requirements.txt          # Required Python libraries
│── README.md                 # Project documentation
```

---

## **🔧 Installation**  
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/your-repo/Contrastive-Learning-HSI.git
cd Contrastive-Learning-HSI
```
  
2️⃣ **Create a virtual environment (optional, but recommended)**  
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
```

3️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

---

## **🚀 Training the Model**  
Run the following command to train the model using contrastive learning:  
```bash
python train.py --epochs 100 --batch_size 64 --lr 1e-4
```
Arguments:
- `--epochs` → Number of training epochs (default: `100`)
- `--batch_size` → Batch size for training (default: `64`)
- `--lr` → Learning rate (default: `1e-4`)  

---

## **📊 Results & Visualizations**  
After training, the results (loss curves, embeddings) will be saved in the `results/` directory.  
To visualize training loss:
```bash
python plot_loss.py
```

---

## **📌 Key Features & Benefits**  
✅ **Self-supervised learning** → No need for large labeled datasets.  
✅ **Superpixel-based local feature extraction** → Improved spatial-spectral feature representation.  
✅ **Contrastive learning framework** → Robust feature learning with positive-negative sample pairs.  
✅ **Flexible architecture** → Can be adapted to various hyperspectral datasets.  

---

## **📚 Citation**  
If you find this work useful, please consider citing it:  
```
@article{your_paper_2024,
  title={Contrastive Learning-Based Hyperspectral Image Classification},
  author={Your Name, Collaborator Name},
  journal={Arxiv Preprint},
  year={2024}
}
```

---

## **📩 Contact**  
For questions or collaboration opportunities, feel free to reach out:  
📧 Email: your_email@example.com  
📌 GitHub: [your-repo](https://github.com/your-repo)  

---

🎯 **Let's build a more effective self-supervised learning framework for hyperspectral image classification! 🚀**  

---

💡 **Would you like to add specific dataset details, visualization examples, or an evaluation script? Let me know, and I can further refine the README!** 🚀
