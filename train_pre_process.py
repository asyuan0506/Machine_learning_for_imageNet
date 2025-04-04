import os
import tarfile
from tqdm import tqdm

'''
這段程式碼的功能是將指定資料夾(ImageNet train images)中的所有 tar 檔案解壓縮到對應的子資料夾中，並在解壓縮後刪除原始的 tar 檔案。
'''

src_dir = "../ImageNet/train_t12_raw" # 來源資料夾
dst_dir = "../ImageNet/train_t12_ready" # 目標資料夾
os.makedirs(dst_dir, exist_ok=True)

for tar_name in tqdm(os.listdir(src_dir)):
    if not tar_name.endswith(".tar"):
        continue
    synset = tar_name.replace(".tar", "")
    synset_dir = os.path.join(dst_dir, synset)
    os.makedirs(synset_dir, exist_ok=True)

    tar_path = os.path.join(src_dir, tar_name)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=synset_dir)
    os.remove(tar_path)

        
