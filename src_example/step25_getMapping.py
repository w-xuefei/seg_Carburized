import os
import pandas as pd
import hashlib
from pathlib import Path
from tqdm import tqdm

# 代码1相关函数
def calculate_image_hash(image_path):
    """计算图片的哈希值（基于文件内容）"""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def find_common_images(path1, path2):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    def build_hash_dict(path):
        hash_dict = {}
        for entry in os.scandir(path):
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in image_extensions:
                    hash_value = calculate_image_hash(entry.path)
                    hash_dict[hash_value] = entry.name
        return hash_dict
    
    images1 = build_hash_dict(path1)
    images2 = build_hash_dict(path2)
    
    return [(images1[hash_val], images2[hash_val]) for hash_val in images1 if hash_val in images2]

def split_filename(filename, delimiter="_"):
    name_without_ext = os.path.splitext(filename)[0]
    if delimiter == "_":
        return name_without_ext.split("_")[0]
    elif delimiter == "-":
        parts = name_without_ext.split("-")
        return parts[0], parts[1] if len(parts) > 1 else None

def create_hardness_mapping(common_images):
    hardness_mapping = {}
    for img1, img2 in common_images:
        hardness_name = split_filename(img2, "_")
        hardness_mapping[img1] = hardness_name
    return hardness_mapping

# 代码2相关函数
def read_hv_data(cr_name: str, process_num: int, data_root: str, n_rows=14):
    folder_name = cr_name.lower()
    formatted_num = f"{process_num:02d}"
    filename = f"process{formatted_num}_1.xls"
    file_path = Path(data_root) / folder_name / filename
    
    if not file_path.exists():
        return []
    
    try:
        df = pd.read_excel(file_path, engine="xlrd")
        if "HV" not in df.columns:
            return []
        hv_values = df["HV"].head(n_rows).tolist()
        return hv_values[:n_rows]
    except Exception as e:
        return []

# 主处理流程
def generate_mapping_csv(output_path, data_root):
    # 初始化路径
    path1 = "/home/ub22/doc/code/seg_cr13/datasets/raw/images"
    path2 = "/home/ub22/doc/code/MHTR-Agent/datasets/dataML/metallographic"
    
    # 获取映射关系
    common_images = find_common_images(path1, path2)
    hardness_mapping = create_hardness_mapping(common_images)
    
    # 准备数据容器
    results = []
    
    # 处理每个图像
    for img1, hardness_name in tqdm(hardness_mapping.items(), desc="处理图像"):
        # 解析硬度名称
        if "-" not in hardness_name:
            continue
            
        try:
            cr_name, process_num = hardness_name.split("-")
            process_num = int(process_num)
        except (ValueError, TypeError):
            continue
        
        # 读取HV数据
        hv_data = read_hv_data(cr_name, process_num, data_root)
        
        # 构建记录
        record = {
            "img1": img1,
            "cr_name": cr_name,
            "process_num": process_num,
            "hv_data": str(hv_data) if hv_data else ""
        }
        results.append(record)
    
    # 生成DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nCSV文件已保存至：{output_path}")

if __name__ == "__main__":
    
    DATA_ROOT = "/home/ub22/doc/code/seg_cr13/datasets/hardness"
    OUTPUT_CSV = "./hardness_mapping.csv"
    
    generate_mapping_csv(OUTPUT_CSV, DATA_ROOT)
    print("示例数据：")
    print(pd.read_csv(OUTPUT_CSV).head(3).to_string(index=False))