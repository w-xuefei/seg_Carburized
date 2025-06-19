"""
目录结构：
输出的目录结构如下：
- images/            存储所有图像文件
- annotations/
  - trimaps/         存储所有处理后的掩码文件
  - trainval.txt     训练和验证集划分文件
  - test.txt         测试集划分文件

作者: Wang
日期: 2025-01-09
"""

import os
import shutil
from PIL import Image
import numpy as np

def construct_dataset(images_dir, masks_dir, output_root):
    """
    构建符合 Oxford Pet 数据集格式的目录结构，并对掩码进行预处理。
    
    参数：
        images_dir (str): 原始图像文件夹路径。
        masks_dir (str): 原始掩码文件夹路径。
        output_root (str): 输出数据集根目录路径。
    """
    # 创建输出目录结构
    images_output_dir = os.path.join(output_root, "images")
    masks_output_dir = os.path.join(output_root, "annotations", "trimaps")
    annotations_dir = os.path.join(output_root, "annotations")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # 获取图像和掩码文件列表
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

    # 确保图像和掩码一一对应
    image_files.sort()
    mask_files.sort()
    print(len(image_files))
    print(len(mask_files))
    assert len(image_files) == len(mask_files), "图像和掩码数量不匹配！"

    # 处理图像和掩码
    for image_file, mask_file in zip(image_files, mask_files):
        # 检查文件名是否一致（去掉扩展名后）
        assert os.path.splitext(image_file)[0] == os.path.splitext(mask_file)[0], \
            f"图像 {image_file} 和掩码 {mask_file} 不匹配！"

        # 复制图像到目标目录
        src_image_path = os.path.join(images_dir, image_file)
        dst_image_path = os.path.join(images_output_dir, image_file)
        shutil.copy(src_image_path, dst_image_path)

        # 复制mask到目标目录
        src_mask_path = os.path.join(masks_dir, mask_file)
        dst_mask_path = os.path.join(masks_output_dir, mask_file)
        shutil.copy(src_mask_path, dst_mask_path)

    # 自动生成 trainval.txt 和 test.txt（简单划分）
    filenames = [os.path.splitext(f)[0] for f in image_files]
    trainval_split = int(len(filenames) * 0.8)  # 80% 训练/验证，20% 测试
    trainval_filenames = filenames[:trainval_split]
    test_filenames = filenames[trainval_split:]

    with open(os.path.join(annotations_dir, "trainval.txt"), "w") as f:
        f.write("\n".join(trainval_filenames))

    with open(os.path.join(annotations_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_filenames))

    print(f"数据集构建完成，已保存到 {output_root}")


# 示例用法
if __name__ == "__main__":

    # 原始图像和掩码文件夹路径
    fixed_image_folder = "../datasets/fixed/images"
    fixed_mask_folder = "../datasets/fixed/masks"

    # 输出数据集根目录
    output_root = "../datasets/output"  # 替换为实际输出路径

    # 构建数据集
    construct_dataset(fixed_image_folder, fixed_mask_folder, output_root)
