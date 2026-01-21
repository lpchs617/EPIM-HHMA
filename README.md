# HH&MA模型：基于异质超图和多尺度注意力的关键蛋白质识别

## 一、模型简介
HH&MA（Heterogeneous Hypergraph & Multi-scale Attention）是一种用于关键蛋白质识别的深度学习模型，通过整合PPI网络、氨基酸序列、基因表达多源数据，构建异质超图刻画蛋白质高阶关联，结合并行多尺度注意力机制学习局部与全局特征，实现高精度关键蛋白质预测。

## 二、环境依赖
```bash
# 基础环境
python >= 3.8
pytorch >= 1.10.0
cuda >= 11.3（可选，GPU加速）

# 依赖库安装
pip install pandas numpy scipy scikit-learn networkx biopython mclustering dtw-python