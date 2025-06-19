"""
输入：
- 二值化掩码图像（灰度图像）。

输出：
- Labelme JSON 文件，包含每个轮廓的多边形数据。

作者: Wang
日期: 2025-01-09
"""

import cv2
import json
import numpy as np

def contours_to_labelme_json(image_path, contours, output_json_path):
    # 读取原始图像以获取图像信息
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # 构造 labelme 格式的 JSON 数据结构
    labelme_data = {
        "version": "5.6.0",  # labelme版本，可以调整为你的版本
        "flags": {},
        "shapes": [],
        "imagePath": image_path.split('/')[-1],
        "imageHeight": height,
        "imageWidth": width,
        "imageData": None  # 不包含 base64 图像数据，labelme 会自动处理
    }

    # 遍历每个轮廓，将其转换为多边形
    for contour in contours:
        # 计算轮廓的面积，过滤掉面积小于4像素的轮廓
        area = cv2.contourArea(contour)
        if area < 3:
            continue  # 跳过面积小于4的轮廓

        # 将 contour 转换为多边形点集 (x, y)
        points = contour.squeeze().tolist()  # 将 numpy 数组转为列表

        shape_data = {
            "label": "111",  # 每个多边形的标签，默认设为 "contour"
            "points": points,  # 轮廓点
            "group_id": None,
            "shape_type": "polygon",  # 形状类型为多边形
            "flags": {}
        }
        labelme_data["shapes"].append(shape_data)

    # 保存为 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(labelme_data, f, indent=4)

    print(f"Labelme JSON file saved to {output_json_path}")

# 主程序
def main():
    # 输入图像路径
    mask_path = "../datasets/test/image_001_mask.jpg"

    # 输出 JSON 文件路径
    output_json_path = "../datasets/test/image_001_labelme.json"

    # 加载图像并转换为灰度图
    image = cv2.imread(mask_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 对灰度图像进行二值化处理，使用 Otsu's 阈值方法
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 将轮廓保存为 Labelme JSON
    contours_to_labelme_json(mask_path, contours, output_json_path)

if __name__ == "__main__":
    main()
