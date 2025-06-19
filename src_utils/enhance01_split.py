import os
from PIL import Image

# 切割尺寸
tile_size = 256

input_list = ["../datasets/raw/images", 
              "../datasets/raw/masks"]
output_list = ["../datasets/split/images", 
               "../datasets/split/masks"]

for i in range(len(input_list)):
    input_dir = input_list[i]
    output_dir = output_list[i]

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 遍历图像文件夹
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            # 读取图像
            img_path = os.path.join(input_dir, img_name)
            image = Image.open(img_path)

            # 获取图像尺寸
            img_width, img_height = image.size

            # 计算行列数量
            cols = img_width // tile_size
            rows = img_height // tile_size

            # 开始切割
            for row in range(rows):
                for col in range(cols):
                    # 定义切割区域
                    left = col * tile_size
                    upper = row * tile_size
                    right = left + tile_size
                    lower = upper + tile_size

                    # 裁剪图像
                    tile = image.crop((left, upper, right, lower))

                    # 保存切割后的图像
                    if i == 0:
                        tile_name = f"{os.path.splitext(img_name)[0]}_r{row}_c{col}.jpg"
                        tile.save(os.path.join(output_dir, tile_name))
                        print(f"切割后的图像已保存为： {output_dir + tile_name}")
                    elif i == 1:
                        tile_name = f"{os.path.splitext(img_name)[0]}_r{row}_c{col}.png"
                        tile.save(os.path.join(output_dir, tile_name))
                        print(f"切割后的图像已保存为： {output_dir + tile_name}")

print("-"*40)
print(f"图像切割完成！切割后的图像已保存到 {output_dir}")
