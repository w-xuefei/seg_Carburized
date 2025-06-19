import os
import cv2
import numpy as np
from PIL import Image

def resize_image(image, target_size=(32, 32)):
    """
    将图像裁剪或填充，使其高度和宽度都能被32整除。
    """
    height, width, channels = image.shape
    new_height = (height // 32) * 32
    new_width = (width // 32) * 32
    
    # 裁剪图像
    if new_height < height:
        image = image[:new_height, :]
    if new_width < width:
        image = image[:, :new_width]
    
    return image

def process_mask(mask, target_size):
    """
    处理掩码，确保其值只为0和1，并且调整大小与目标图像一致。
    """
    # 调整掩码大小到目标尺寸
    mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # # 将掩码中的非0值（例如255）转换为1
    # mask[mask != 0] = 1
    return mask

def check_and_process_image(image_path, save_path):
    """
    检查并处理图像：调整尺寸并保存到指定路径。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 调整图像尺寸
    image = resize_image(image)
    
    # 保存处理后的图像
    cv2.imwrite(save_path, image)
    return image

def check_and_process_mask(mask_path, expected_size, save_fixed_mask=False, output_path=None):
    """
    检查并处理掩码：调整大小、处理值并保存到指定路径。
    """
    # 读取掩码
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 加载掩码图像并转换为灰度图
    mask = np.array(Image.open(mask_path).convert("L"))

    mask = process_mask(mask, expected_size)

    # 初始化结果
    result = {
        "is_valid": True,  # 是否符合要求
        "issues": [],  # 检查发现的问题
        "suggestions": []  # 修复建议
    }

    # Step 1: 对掩码图像进行二值化
    binary_mask = (mask <= 128).astype(np.uint8)  # 大于等于阈值为 1，其余为 0
    unique_values = np.unique(binary_mask)

    # 检查是否为二值图
    if not set(unique_values).issubset({0, 1}):
        result["is_valid"] = False
        result["issues"].append("掩码图像未正确二值化。")
        result["suggestions"].append("对掩码图像进行二值化处理，将像素值调整为 {0, 1}。")

    # Step 2: 检查二值化后的掩码是否符合 Oxford Pet 格式
    # Oxford Pet 数据集要求掩码的像素值为 {0, 1}，其中 0 表示背景，1 表示目标
    if not set(unique_values).issubset({0, 1}):
        result["is_valid"] = False
        result["issues"].append("掩码图像像素值不符合 {0, 1} 格式。")
        result["suggestions"].append("将像素值调整为符合 Oxford Pet 数据集的标准格式。")
    else:
        result["issues"].append("掩码图像已经符合 Oxford Pet 数据集格式。")

    # Step 3: 保存修复后的掩码图像（如果需要修复）
    if save_fixed_mask:
        fixed_mask = binary_mask

        # 保存修复后的掩码
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray((fixed_mask * 1).astype(np.uint8)).save(output_path)
            result["suggestions"].append(f"修复后的掩码已保存到: {output_path}")

    return result

def process_dataset(image_folder, mask_folder, fixed_image_folder, fixed_mask_folder):
    # 获取图像和掩码文件列表
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    if len(image_files) != len(mask_files):
        print("错误: 图像和掩码文件数量不匹配")
        return

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        # 生成保存路径
        fixed_image_path = os.path.join(fixed_image_folder, image_file)
        fixed_mask_path = os.path.join(fixed_mask_folder, mask_file.replace('.jpg', '.png'))  # 将掩码保存为PNG格式

        print(f"处理 {image_path} 和 {mask_path}:")

        # 处理图像
        image = check_and_process_image(image_path, fixed_image_path)
        if image is None:
            continue

        # 处理掩码
        result = check_and_process_mask(mask_path, image.shape[:2],save_fixed_mask=True, output_path=fixed_mask_path)
        print("mask检查结果:", result)
        # if mask is None:
        #     continue

        print(f"已处理并保存 {fixed_image_path} 和 {fixed_mask_path}")

    print("数据集处理完成！")

if __name__ == "__main__":
    
    image_folder = "../datasets/augmented/images"
    mask_folder = "../datasets/augmented/masks"
    fixed_image_folder = "../datasets/fixed/images"
    fixed_mask_folder = "../datasets/fixed/masks"

    # 创建目标文件夹（如果不存在的话）
    os.makedirs(fixed_image_folder, exist_ok=True)
    os.makedirs(fixed_mask_folder, exist_ok=True)

    # 处理数据集
    process_dataset(image_folder, mask_folder, fixed_image_folder, fixed_mask_folder)
