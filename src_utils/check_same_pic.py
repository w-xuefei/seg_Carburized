import os

def compare_image_folders(folder1, folder2):
    # 获取两个文件夹中的所有文件名
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    
    # 提取文件名（去除扩展名）
    filenames1 = {os.path.splitext(f)[0] for f in files1}
    filenames2 = {os.path.splitext(f)[0] for f in files2}
    
    # 比较两个文件夹中文件名的名称
    common_images = filenames1 & filenames2
    only_in_folder1 = filenames1 - filenames2
    only_in_folder2 = filenames2 - filenames1
    
    return common_images, only_in_folder1, only_in_folder2

# 示例使用
folder1 = "../datasets/augmented/images"
folder2 = "../datasets/augmented/masks"
              
common, only_in_folder1, only_in_folder2 = compare_image_folders(folder1, folder2)

# print(f"相同名称的图片：{common}")
print(f"仅在文件夹1中的图片：{only_in_folder1}")
print(f"仅在文件夹2中的图片：{only_in_folder2}")
