import os
import shutil

# Paths
cub_root    = "CUB_200_2011"
images_root = os.path.join(cub_root, "images")
ann_root    = cub_root          # <— look in the top-level folder, not “annotations”
out_train   = "data/train"
out_valid   = "data/valid"


# 1) Map image IDs to relative paths
id_to_path = {}
with open(os.path.join(ann_root, "images.txt")) as f:
    for line in f:
        img_id, rel_path = line.strip().split()
        id_to_path[img_id] = rel_path

# 2) Read the official train/test split flags
split_flags = {}
with open(os.path.join(ann_root, "train_test_split.txt")) as f:
    for line in f:
        img_id, is_train = line.strip().split()
        split_flags[img_id] = (is_train == "1")

# 3) Copy each image into the proper folder
for img_id, rel_path in id_to_path.items():
    src = os.path.join(images_root, rel_path)
    species = rel_path.split("/")[0]
    dest_root = out_train if split_flags[img_id] else out_valid
    dest_dir = os.path.join(dest_root, species)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(src, dest_dir)

