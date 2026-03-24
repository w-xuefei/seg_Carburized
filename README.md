# seg_Carburized

This repository contains the official code and dataset for the paper **Multimodal learning-guided performance prediction and design of carburized steels via deep microstructural characterization** (Surface & Coatings Technology, 2025, 517: 132861). The work proposes an integrated multimodal learning framework for hardness prediction and carburizing process optimization of carburized steels, combining deep semantic segmentation for carbide phase detection, machine learning for hardness prediction, and SHAP analysis for feature interpretability.

## Paper Information

- **Authors**: Xuefei Wang, Chunyang Luo, Di Jiang, Yihao Zheng, Haojie Wang, Chi Zhang, Zhaodong Wang*

- **Journal**: Surface & Coatings Technology

- **DOI**: [https://doi.org/10.1016/j.surfcoat.2025.132861](https://doi.org/10.1016/j.surfcoat.2025.132861)

- **Affiliations**: Northeastern University, Shenyang University of Technology, Henan Academy of Sciences

## Key Contributions

1. Integrate deep semantic segmentation, feature extraction and machine learning to realize high-precision hardness prediction of carburized steels.

2. Propose **UNet++ with DenseNet121** as the optimal model for pixel-level segmentation of carbide phases in metallographic images.

3. Use **SHAP analysis** to quantify the contribution of alloy composition, process parameters and microstructural features to hardness, identifying key influencing factors (Cr content, carbide distribution, heat treatment parameters).

4. Construct a high-quality dataset with **500+ entries** covering multiple steel grades and 20 carburizing processes, based on LLM-assisted unstructured data extraction.

5. Achieve state-of-the-art hardness prediction performance with **MLP model (R²=0.996, RMSE=1.95)**.

![alt text](<images/Graphical Abstract‌‌.png>)

## Citation

If you use the code, dataset or results of this work in your research, please cite our paper:

```Plain Text

@article{WANG2025132861,
  title = {Multimodal learning-guided performance prediction and design of carburized steels via deep microstructural characterization},
  journal = {Surface & Coatings Technology},
  volume = {517},
  pages = {132861},
  year = {2025},
  issn = {0257-8972},
  doi = {https://doi.org/10.1016/j.surfcoat.2025.132861},
  author = {Xuefei Wang and Chunyang Luo and Di Jiang and Yihao Zheng and Haojie Wang and Chi Zhang and Zhaodong Wang},
  keywords = {Carburizing, Hardness prediction, Semantic segmentation, Carbide morphology, SHAP}
}
```

## Acknowledgments

This work was supported by:

1. High-level Talent Research Start-up Project Funding of Henan Academy of Sciences (No.242017003)

2. Joint Fund of Henan Province Science and Technology R&D Program (No.225200810040)

We thank the open-source communities of PyTorch, OpenCV, scikit-learn and SHAP for their excellent tools. We also acknowledge the use of DeepSeek and GPT for LLM-assisted unstructured data extraction.

