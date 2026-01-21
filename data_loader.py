import pandas as pd
import numpy as np
from Bio import SeqIO
from config import config

class DataLoader:
    def __init__(self):
        self.ppi_data = None
        self.gene_exp_data = None
        self.sequence_data = None
        self.essential_labels = None
        self.protein_ids = None  # 统一蛋白质ID列表

    def load_ppi(self):
        """加载PPI网络数据（格式：protein1 protein2）"""
        ppi = pd.read_csv(config.PPI_DATA_PATH, sep="\t", header=None, names=["p1", "p2"])
        # 去重、去除自环
        ppi = ppi[ppi["p1"] != ppi["p2"]].drop_duplicates()
        self.ppi_data = ppi
        return ppi

    def load_gene_expression(self):
        """加载基因表达数据（行：蛋白质，列：时间点）"""
        expr = pd.read_csv(config.GENE_EXP_PATH, index_col=0)
        self.gene_exp_data = expr
        return expr

    def load_sequences(self):
        """加载氨基酸序列数据（FASTA格式）"""
        sequences = {}
        for record in SeqIO.parse(config.SEQUENCE_PATH, "fasta"):
            protein_id = record.id.split("|")[0]  # 提取蛋白质ID
            sequences[protein_id] = str(record.seq)
        self.sequence_data = sequences
        return sequences

    def load_essential_labels(self):
        """加载关键蛋白质标签（格式：protein_id 1/0）"""
        labels = pd.read_csv(config.ESSENTIAL_LABEL_PATH, sep="\t", header=None, names=["protein", "label"])
        self.essential_labels = labels.set_index("protein")["label"].to_dict()
        return self.essential_labels

    def get_unified_proteins(self):
        """获取所有数据源的交集蛋白质ID"""
        ppi_proteins = set(self.ppi_data["p1"]).union(set(self.ppi_data["p2"]))
        expr_proteins = set(self.gene_exp_data.index)
        seq_proteins = set(self.sequence_data.keys())
        label_proteins = set(self.essential_labels.keys())
        
        # 取交集
        self.protein_ids = list(ppi_proteins & expr_proteins & seq_proteins & label_proteins)
        self.protein_ids.sort()  # 排序保证一致性
        print(f"统一后蛋白质数量：{len(self.protein_ids)}")
        return self.protein_ids

    def run(self):
        """执行所有数据加载流程"""
        print("加载PPI网络...")
        self.load_ppi()
        print("加载基因表达数据...")
        self.load_gene_expression()
        print("加载氨基酸序列...")
        self.load_sequences()
        print("加载关键蛋白质标签...")
        self.load_essential_labels()
        print("获取统一蛋白质列表...")
        self.get_unified_proteins()
        return self