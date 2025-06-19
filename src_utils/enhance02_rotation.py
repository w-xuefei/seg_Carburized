import os
from PIL import Image


input_list = ["../datasets/split/images", 
              "../datasets/split/masks"]
output_list = ["../datasets/augmented/images", 
               "../datasets/augmented/masks"]

for i in range(len(input_list)):
    image_dir = input_list[i]
    aug_image_dir = output_list[i]

    # 创建输出目录
    os.makedirs(aug_image_dir, exist_ok=True)

    # 定义旋转角度
    angles = [90, 180, 270]

    # 遍历图像文件夹
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(image_dir, img_name)

            # 打开图像和掩码
            image = Image.open(img_path)

            # 保存原图像和掩码
            image.save(os.path.join(aug_image_dir, img_name))

            # 对图像和掩码进行旋转增强
            for angle in angles:
                # 旋转图像和掩码
                rotated_image = image.rotate(angle, expand=True)
                # 保存旋转后的图像和掩码
                if i == 0:
                    rotated_image_name = f"{os.path.splitext(img_name)[0]}_rot{angle}.jpg"
                    rotated_image.save(os.path.join(aug_image_dir, rotated_image_name))
                    print(f"旋转后的image已保存为： {aug_image_dir + rotated_image_name}")
                elif i == 1:
                    rotated_image_name = f"{os.path.splitext(img_name)[0]}_rot{angle}.png"
                    rotated_image.save(os.path.join(aug_image_dir, rotated_image_name))
                    print(f"旋转后的image已保存为： {aug_image_dir + rotated_image_name}")

print(f"旋转数据增强完成！增强后的图像已保存到 {aug_image_dir}")