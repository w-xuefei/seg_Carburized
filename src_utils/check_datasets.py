"""
说明：
    - 检查指定文件夹中的图像和掩码是否符合要求：
    - 图像是否具有相同的大小。
    - 图像的高度和宽度是否能被32整除。
    - 图像是否具有正确的轴顺序（CHW格式）。
    - 掩码是否与图像的大小相同。
    - 掩码是否只包含0和1。
    - 掩码是否需要转换为1通道。
"""

import os
import cv2
import numpy as np

def check_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 检查图像尺寸
    height, width, channels = image.shape
    print("当前图片的尺寸为：", height, width, channels)
    if height % 32 != 0 or width % 32 != 0:
        print(f"警告: 图像 {image_path} 的尺寸 {height}x{width} 不能被32整除")
    
    # 检查图像轴顺序 (HWC -> CHW)
    image_chw = np.transpose(image, (2, 0, 1))  # 将图像转换为CHW格式
    return image_chw, (height, width)

def check_mask(mask_path, expected_size):
    # 读取掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"无法读取掩码: {mask_path}")
        return None

    # 检查掩码大小
    height, width = mask.shape
    print("当前掩码的尺寸为：", height, width)
    if (height, width) != expected_size:
        print(f"警告: 掩码 {mask_path} 的尺寸 {height}x{width} 与图像尺寸不匹配")
    
    # 检查掩码的值是否为0或1
    unique_values = np.unique(mask)
    if not np.all(np.isin(unique_values, [0, 1])):
        print(f"警告: 掩码 {mask_path} 包含除0和1之外的值: {unique_values}")
    else:
        print("掩码的值为0或1:", unique_values)
    
    # 检查掩码是否有正确的通道数 (扩展成单通道)
    if len(mask.shape) == 2:  # 单通道
        mask = np.expand_dims(mask, axis=2)  # 扩展为1通道
    return mask

def check_dataset(image_folder, mask_folder):

    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    print(len(image_files))
    print(len(mask_files))

    if len(image_files) != len(mask_files):
        print("错误: 图像和掩码文件数量不匹配")
        return

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        print(f"检查 {image_path} 和 {mask_path}:")

        # 检查图像
        image_chw, image_size = check_image(image_path)
        if image_chw is None:
            continue

        # 检查掩码
        mask = check_mask(mask_path, image_size)
        if mask is None:
            continue

    print("数据集检查完成！")

if __name__ == "__main__":
    
    image_folder = "../datasets/split/images"
    mask_folder = "../datasets/split/masks"

    check_dataset(image_folder, mask_folder)
