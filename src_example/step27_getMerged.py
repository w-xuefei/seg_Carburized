import pandas as pd
import numpy as np

# 预定义的映射字典
grade_dict = {
    '1cr': ['0.08', '1.00', '1.00', '0.04', '0.03', '0.06', '11.50', '86.29'],
    '2cr': ['0.16', '1.00', '1.00', '0.04', '0.03', '0.06', '13.00', '84.71'],
    '3cr': ['0.26', '1.00', '1.00', '0.04', '0.03', '0.06', '12.00', '85.61'],
    '4cr': ['0.36', '0.60', '0.80', '1.04', '0.03', '0.06', '13.00', '84.11']
}

process_dict = {
    'process_01': ['1050', '1000', '160'],
    'process_02': ['1050', '1000', '200'],
    'process_03': ['1050', '1000', '240'],
    'process_04': ['1050', '1020', '160'],
    'process_05': ['1050', '1020', '200'],
    'process_06': ['1050', '1020', '240'],
    'process_07': ['1050', '1040', '160'],
    'process_08': ['1050', '1040', '200'],
    'process_09': ['1050', '1040', '240'],
    'process_10': ['980', '1000', '160'],
    'process_11': ['980', '1000', '200'],
    'process_12': ['980', '1000', '240'],
    'process_13': ['980', '1020', '160'],
    'process_14': ['980', '1020', '200'],
    'process_15': ['980', '1020', '240'],
    'process_16': ['980', '1040', '160'],
    'process_17': ['980', '1040', '200'],
    'process_18': ['980', '1040', '240']
}


# Density (g/cm3), Electrical Conductivity (Ohm*m), Thermal Conductivity (W/(m·K)), 
# Bulk Modulus (GPa), Young's Modulus (GPa), Specific Heat Capacity (J/(kg·℃))
parameters_dict = {
    '1cr': ['7.7', '0.0000018', '17.46', '170', '213.58', '0.45'],
    '2cr': ['7.69', '0.0000018', '17.48', '169.89', '214.45', '0.45'],
    '3cr': ['7.68', '0.0000018', '17.49', '169.71', '215.27', '0.46'],
    '4cr': ['7.66', '0.0000018', '17.52', '169.5', '216.06', '0.46']
}

def merge_features_hardness(features_path, mapping_path, output_path):
    # 读取数据
    features_df = pd.read_csv(features_path)
    mapping_df = pd.read_csv(mapping_path)
    
    # 处理文件名基准名
    features_df["base_name"] = features_df["Image_Name"].str.replace(r"\..*", "", regex=True)
    mapping_df["base_name"] = mapping_df["img1"].str.replace(r"\..*", "", regex=True)
    
    # 合并数据集（内连接）
    merged_df = pd.merge(
        features_df,
        mapping_df[["base_name", "cr_name", "process_num", "hv_data"]],
        on="base_name",
        how="inner"  # 改为内连接自动过滤空值
    ).drop(columns=["base_name"])
    
    # 类型转换
    merged_df["process_num"] = merged_df["process_num"].astype(int)
    
    # 映射材料成分数据 --------------------------------------------------------------------------------------------------------------
    merged_df["cr_lower"] = merged_df["cr_name"].str.lower()
    merged_df = merged_df.merge(
        pd.DataFrame.from_dict(grade_dict, orient='index', 
                              columns=['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Fe']),
        left_on="cr_lower",
        right_index=True
    ).drop(columns=["cr_lower"])
    
    # 映射工艺参数数据 -------------------------------------------------------
    merged_df["process_key"] = "process_" + merged_df["process_num"].apply(lambda x: f"{x:02d}")
    merged_df = merged_df.merge(
        pd.DataFrame.from_dict(process_dict, orient='index',
                              columns=['carburizing_temp', 'quenching_temp', 'tempering_temp']),
        left_on="process_key",
        right_index=True
    ).drop(columns=["process_key"])

    # 映射材料物理特性数据 -------------------------------------------------------
    merged_df["cr_lower"] = merged_df["cr_name"].str.lower()
    merged_df = merged_df.merge(
        pd.DataFrame.from_dict(parameters_dict, orient='index', 
                              columns=["Density_phy", "ElectricalConductivity", "ThermalConductivity","BulkModulus", "YoungsModulus", "SpecificHeatCapacity"]),
        left_on="cr_lower",
        right_index=True
    ).drop(columns=["cr_lower"])
    
    # 保存结果
    merged_df.to_csv(output_path, index=False)
    print(f"合并完成，有效记录数：{len(merged_df)}")
    return merged_df

if __name__ == "__main__":
    # 路径配置
    FEATURES_CSV = "./image_features.csv"
    MAPPING_CSV = "./hardness_mapping.csv"
    OUTPUT_CSV = "./merged_dataset.csv"
    
    # 执行合并
    result_df = merge_features_hardness(FEATURES_CSV, MAPPING_CSV, OUTPUT_CSV)
    
    # 显示结果示例
    print("\n合并结果示例：")
    print(result_df.head(3).to_string(index=False))