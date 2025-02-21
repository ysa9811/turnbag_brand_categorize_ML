import requests
import pymysql
import requests
from io import BytesIO
import pandas as pd
import matplotlib.image as mpimg
from concurrent.futures import ThreadPoolExecutor
import os
from PIL import Image
import numpy as np

# 이미지가 저장된 루트 폴더 경로
root_dir = r"C:\Users\user\Desktop\YOLO"

def fetch_local_images(base_path):
    data = []
    for brand_folder in os.listdir(base_path):
        brand_path = os.path.join(base_path, brand_folder)
        if os.path.isdir(brand_path):
            for image_file in os.listdir(brand_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(brand_path, image_file)
                    try:
                        image = Image.open(image_path).convert("RGB")  # RGB로 변환
                        image_array = np.array(image)
                        data.append({
                            "idx": None, 
                            "brand_name": brand_folder,
                            "image_rgb": image_array,  
                        })
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
    return pd.DataFrame(data)

def mysql():
    local_df = fetch_local_images(root_dir)
    brand_counts = local_df['brand_name'].value_counts()
    print(brand_counts)
    return local_df