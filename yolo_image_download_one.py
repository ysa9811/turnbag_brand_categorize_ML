import os
from PIL import Image
from ml_image_data import mysql

def download_images_and_save(df, base_path, target_brands):
    """
    특정 브랜드의 이미지만 다운로드하여 저장하는 함수

    :param df: 이미지 데이터프레임
    :param base_path: 이미지 저장 경로
    :param target_brands: 다운로드할 브랜드 이름 목록 (리스트), None이면 전체 저장
    """
    # 이미지 저장할 기본 경로 설정
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # 특정 브랜드만 필터링 (target_brands가 제공되었을 때만 필터링)
    if target_brands:
        df = df[df['brand_name'].isin(target_brands)]
    
    # 브랜드별 폴더 생성 및 이미지 저장
    for _, row in df.iterrows():
        brand_name = str(row['brand_name'])  # 브랜드 이름을 문자열로 변환
        image_rgb = row['image_rgb']
        
        # 브랜드 폴더가 없다면 생성
        brand_folder = os.path.join(base_path, brand_name)
        if not os.path.exists(brand_folder):
            os.makedirs(brand_folder)
        
        # 이미지 파일명 설정 (idx를 브랜드 이름과 함께 사용하는 방식으로 변경)
        image_filename = f"{row['idx'] if row['idx'] else 'unknown'}.jpg"
        image_path = os.path.join(brand_folder, image_filename)
        
        try:
            # 이미지를 파일로 저장
            img = Image.fromarray(image_rgb)
            img.save(image_path)
            print(f"Saved image: {image_path}")
        except Exception as e:
            print(f"Error saving image for {brand_name} ({row['idx']}): {e}")

# 저장할 경로
base_path = r"C:\Users\user\Desktop\YOLO"

# 원하는 브랜드 목록 (예: Gucci와 Prada만 다운로드)
target_brands = [7]

# 실행
download_images_and_save(mysql(), base_path, target_brands)
