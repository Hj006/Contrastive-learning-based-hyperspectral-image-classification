# datasets.py
import numpy as np
import random
from pathlib import Path
from scipy.io import loadmat
from scipy.sparse import coo_matrix
import skimage.io
import rasterio
from io import StringIO
import warnings
import torch
from torch.utils.data import Dataset

def load_pavia_university_with_full_test(data_path: Path, candidate_counts: dict, num_train_per_class=15, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    pavia_data = loadmat(data_path / 'PaviaU.mat')['paviaU']
    gt = loadmat(data_path / 'PaviaU_gt.mat')['paviaU_gt']
    h, w, c = pavia_data.shape
    label_dict = {
        1: 'Asphalt road', 2: 'Meadows', 3: 'Gravel',
        4: 'Trees', 5: 'Painted metal sheets', 6: 'Bare Soil',
        7: 'Bitumen', 8: 'Self-Blocking Bricks', 9: 'Shadows',
    }
    train_rows, train_cols, train_data = [], [], []
    candidate_rows, candidate_cols, candidate_data = [], [], []
    for label, cand_count in candidate_counts.items():
        coords = np.argwhere(gt == label)
        np.random.shuffle(coords)
        candidate_coords = coords[:cand_count]
        train_coords = candidate_coords[:num_train_per_class]
        for r, c in train_coords:
            train_rows.append(r)
            train_cols.append(c)
            train_data.append(label)
        for r, c in candidate_coords:
            candidate_rows.append(r)
            candidate_cols.append(c)
            candidate_data.append(label)
    train_truth = coo_matrix((train_data, (train_rows, train_cols)), shape=(h, w), dtype=int)
    candidate_truth = coo_matrix((candidate_data, (candidate_rows, candidate_cols)), shape=(h, w), dtype=int)
    test_rows, test_cols = np.where(gt > 0)
    test_data = gt[test_rows, test_cols]
    test_truth = coo_matrix((test_data, (test_rows, test_cols)), shape=(h, w), dtype=int)
    info = {'n_band': c, 'width': w, 'height': h, 'label_dict': label_dict}
    return pavia_data.transpose(2, 0, 1), train_truth, candidate_truth, test_truth, info

def load_whu_longkou(data_path: Path, candidate_counts: dict, num_train_per_class=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    with rasterio.open(data_path / "WHU-Hi-LongKou.tif") as src:
        hyperspectral = src.read()
    c, h, w = hyperspectral.shape
    with rasterio.open(data_path / "WHU-Hi-LongKou_gt.tif") as src:
        gt = src.read(1)
    label_dict = {
        1: '玉米', 2: '棉花', 3: '芝麻',
        4: '圆叶大豆', 5: '长叶大豆', 6: '水稻',
        7: '水体', 8: '房屋和道路', 9: '混合杂草',
    }
    train_rows, train_cols, train_data = [], [], []
    candidate_rows, candidate_cols, candidate_data = [], [], []
    for label, cand_count in candidate_counts.items():
        coords = np.argwhere(gt == label)
        np.random.shuffle(coords)
        candidate_coords = coords[:cand_count]
        train_coords = candidate_coords[:num_train_per_class]
        for r, c in train_coords:
            train_rows.append(r)
            train_cols.append(c)
            train_data.append(label)
        for r, c in candidate_coords:
            candidate_rows.append(r)
            candidate_cols.append(c)
            candidate_data.append(label)
    train_truth = coo_matrix((train_data, (train_rows, train_cols)), shape=(h, w), dtype=int)
    candidate_truth = coo_matrix((candidate_data, (candidate_rows, candidate_cols)), shape=(h, w), dtype=int)
    test_rows, test_cols = np.where(gt > 0)
    test_data = gt[test_rows, test_cols]
    test_truth = coo_matrix((test_data, (test_rows, test_cols)), shape=(h, w), dtype=int)
    info = {'n_band': c, 'width': w, 'height': h, 'label_dict': label_dict}
    return hyperspectral, train_truth, candidate_truth, test_truth, info

def _read_roi(path: Path, shape):
    warnings.simplefilter("ignore", category=UserWarning)
    data = []
    rows = []
    cols = []
    current_label = 0
    buffer = ""
    with open(path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith(";") or "ROI" in line:
                if buffer:
                    roi_data = np.loadtxt(StringIO(buffer), usecols=(2, 1), dtype=int)
                    if roi_data.size > 0:
                        r, c = roi_data.T
                        rows.extend(r)
                        cols.extend(c)
                        data.extend([current_label] * len(r))
                    buffer = ""
                if "ROI name" in line:
                    current_label += 1
            else:
                buffer += line
        if buffer:
            roi_data = np.loadtxt(StringIO(buffer), usecols=(2, 1), dtype=int)
            if roi_data.size > 0:
                r, c = roi_data.T
                rows.extend(r)
                cols.extend(c)
                data.extend([current_label] * len(r))
    warnings.resetwarnings()
    img = coo_matrix((data, (rows, cols)), shape=shape, dtype=int)
    return img

def sample_train_from_candidate(candidate_truth, num_per_class=15, seed=42):
    np.random.seed(seed)
    rows, cols, labels = candidate_truth.row, candidate_truth.col, candidate_truth.data
    new_rows, new_cols, new_labels = [], [], []
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        indices = np.where(labels == cls)[0]
        chosen = np.random.choice(indices, num_per_class, replace=False)
        new_rows.extend(rows[chosen])
        new_cols.extend(cols[chosen])
        new_labels.extend(labels[chosen])
    shape = candidate_truth.shape
    train_truth = coo_matrix((new_labels, (new_rows, new_cols)), shape=shape, dtype=int)
    return train_truth

def load_houston2013(data_path: Path):
    FILES_PATH = data_path
    lidar = skimage.io.imread(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_LiDAR.tif')[np.newaxis, :, :]
    casi = skimage.io.imread(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_CASI.tif').transpose(2, 0, 1)
    train_truth = _read_roi(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_TR.txt', (349, 1905))
    test_truth = _read_roi(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_VA.txt', (349, 1905))
    info = {
        'n_band_casi': 144,
        'n_band_lidar': 1,
        'width': 1905,
        'height': 349,
        'label_dict': {
            1: 'Healthy grass', 2: 'Stressed grass', 3: 'Synthetic grass', 4: 'Trees',
            5: 'Soil', 6: 'Water', 7: 'Residential', 8: 'Commercial', 9: 'Road',
            10: 'Highway', 11: 'Railway', 12: 'Parking Lot 1', 13: 'Parking Lot 2',
            14: 'Tennis Court', 15: 'Running Track',
        }
    }
    candidate_truth = train_truth
    train_truth = sample_train_from_candidate(candidate_truth, num_per_class=15)
    return casi, train_truth, candidate_truth, test_truth, info

def extract_cube(data, x, y, size):
    c, h, w = data.shape
    half_size = size[0] // 2
    x_min = max(0, x - half_size)
    x_max = min(h, x + half_size + 1)
    y_min = max(0, y - half_size)
    y_max = min(w, y + half_size + 1)
    cube = data[:, x_min:x_max, y_min:y_max]
    pad_width = [
        (0, 0),
        (max(0, half_size - x), max(0, x + half_size + 1 - h)),
        (max(0, half_size - y), max(0, y + half_size + 1 - w)),
    ]
    cube = np.pad(cube, pad_width, mode="reflect")
    return cube

def extract_labels(truth, label_dict):
    rows, cols, labels = truth.row, truth.col, truth.data
    label_to_index = {label_value: idx for idx, label_value in enumerate(label_dict.keys())}
    mapped_labels = [label_to_index[label] for label in labels if label in label_to_index]
    return [(row, col, label) for row, col, label in zip(rows, cols, mapped_labels)]

class NeighborhoodSplitPositiveDataset(Dataset):
    """
    Generate positive pairs by extracting spatial neighbor patches and then splitting
    the spectral channels of the pair for mix-augmentation (half from A, half from B).
    """
    def __init__(self, data, coords, patch_size=11):
        self.data = data
        self.coords = coords
        self.patch_size = patch_size

        # For each center coordinate, generate 4 neighbor pairs
        self.flattened_coords = []
        for x, y in coords:
            self.flattened_coords += [
                (x, y, x-1, y),  # up
                (x, y, x+1, y),  # down
                (x, y, x, y-1),  # left
                (x, y, x, y+1),  # right
            ]

    def __len__(self):
        return len(self.flattened_coords)

    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.flattened_coords[idx]
        patch_a = extract_cube(self.data, x1, y1, (self.patch_size, self.patch_size))
        patch_b = extract_cube(self.data, x2, y2, (self.patch_size, self.patch_size))
        patch_a = torch.tensor(patch_a, dtype=torch.float32)  # [C, H, W]
        patch_b = torch.tensor(patch_b, dtype=torch.float32)  # [C, H, W]

        # Split the channel dimension and mix half from each patch
        C = patch_a.shape[0]
        c_half = C // 2
        a1, a2 = patch_a[:c_half], patch_a[c_half:]
        b1, b2 = patch_b[:c_half], patch_b[c_half:]

        patch_mix1 = torch.cat([a1, b1], dim=0)  # [C, H, W]
        patch_mix2 = torch.cat([a2, b2], dim=0)  # [C, H, W]
        return patch_mix1, patch_mix2


class ClassificationDataset(Dataset):
    def __init__(self, data, labels, patch_size=11):
        self.data = data
        self.labels = labels
        self.patch_size = patch_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x, y, label = self.labels[idx]
        cube = extract_cube(self.data, x, y, (self.patch_size, self.patch_size))
        cube_tensor = torch.tensor(cube).float()
        return cube_tensor, torch.tensor(label, dtype=torch.long)
