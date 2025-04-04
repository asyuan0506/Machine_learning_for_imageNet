import os
import shutil
from scipy.io import loadmat
from tqdm import tqdm

'''
處理 ImageNet validation 圖片資料夾
1. 讀取 meta.mat 檔案，取得 synset 對應的 ID。
2. 讀取 ground truth 檔案，取得每張圖片的 label index。
3. 依據 label index，將圖片複製到對應的 synset 資料夾中。
'''

# 設定路徑
val_raw_dir = "../ImageNet/val_raw" # 原始的 val 圖片檔案資料夾
val_ready_dir = "../ImageNet/val_t12_ready" # 處理後的 val 圖片資料夾
ground_truth_path = "../ImageNet/ILSVRC2012_validation_ground_truth_t12.txt" # ground truth 檔案
meta_path = "../ImageNet/meta_t12.mat" # meta 檔案

# 建立輸出資料夾
os.makedirs(val_ready_dir, exist_ok=True)

# 讀取 meta.mat → ID 對應 synset
meta = loadmat(meta_path)
synsets = meta["synsets"].squeeze()

id_to_synset = {}
for entry in synsets:
    idx = int(entry[0])
    wnid = entry[1][0]
    id_to_synset[idx] = wnid

# 讀取 ground truth（每張圖片的 label index）
with open(ground_truth_path) as f:
    labels = [int(line.strip()) for line in f.readlines()]  # 1-based

# 處理 val 圖
for i in tqdm(range(len(labels))):
    label_id = labels[i]
    synset = id_to_synset.get(label_id)
    if not synset:
        continue

    # 建立目標資料夾
    synset_dir = os.path.join(val_ready_dir, synset)
    os.makedirs(synset_dir, exist_ok=True)

    # 構造檔案名稱（補 8 位數）
    filename = f"ILSVRC2012_val_{i+1:08d}.JPEG"
    src_path = os.path.join(val_raw_dir, filename)
    dst_path = os.path.join(synset_dir, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"⚠️ 找不到圖片：{src_path}")
