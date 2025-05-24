import os
import shutil

PARENT_DIR = "datasets/HeyDocAI_datasets"
TARGET_DIR = "datasets"

if os.path.exists(PARENT_DIR):
    for folder in os.listdir(PARENT_DIR):
        src = os.path.join(PARENT_DIR, folder)
        dst = os.path.join(TARGET_DIR, folder)
        if os.path.isdir(src):
            shutil.move(src, dst)
    shutil.rmtree(PARENT_DIR)
    print("Folder structure flattened successfully.")
else:
    print("Expected nested folder not found.")
