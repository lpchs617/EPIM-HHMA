import numpy as np
import networkx as nx
from mcl_clustering import mcl  # 第三方MCL聚类库
from Bio.Align import PairwiseAligner
from scipy.stats import pearsonr
from config import config
from data_processing.data_loader import DataLoader

class HypergraphConstructor:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.protein2idx = {p: i for i, p in enumerate(data_loader.protein_ids)}  # 蛋白质ID映射为索引
        self.num_proteins = len(data_loader.protein_ids)
        self.hyperedges = []  # 超边列表：每个元素为（超边类型，节点索引集合，权重）
        self.hyperedge_weights = []  # 超边权重

    def construct_ppi_hyperedges(self):
        """构建PPI超边（MCL聚类识别功能模块）"""
        # 构建PPI图
        ppi_graph = nx.Graph()
        ppi_graph.add_nodes_from(self.data_loader.protein_ids)
        valid_edges = self.data_loader.ppi_data[
            (self.data_loader.ppi_data["p1"].isin(self.data_loader.protein_ids)) &
            (self.data_loader.ppi_data["p2"].isin(self.data_loader.protein_ids))
        ]
        ppi_graph.add_edges_from(zip(valid_edges["p1"], valid_edges["p2"]))
        
        # MCL聚类
        adj_matrix = nx.to_numpy_array(ppi_graph, nodelist=self.data_loader.protein_ids)
        clusters = mcl(adj_matrix, inflation=config.MCL_INFLATION)
        
        # 转换为超边（过滤过小的簇）
        for cluster in clusters:
            if len(cluster) >= config.HYPEREDGE_NODE_THRESHOLD:
                node_indices = [self.protein2idx[p] for p in cluster if p in self.protein2idx]
                if len(node_indices) >= config.HYPEREDGE_NODE_THRESHOLD:
                    self.hyperedges.append(("ppi", set(node_indices)))
        print(f"构建PPI超边数量：{len([h for h in self.hyperedges if h[0] == 'ppi'])}")

    def construct_sequence_hyperedges(self):
        """构建序列相似性超边（GSA比对划分同源簇）"""
        aligner = PairwiseAligner()
        aligner.mode = "global"
        sequences = self.data_loader.sequence_data
        proteins = self.data_loader.protein_ids
        seq_clusters = []
        
        # 贪心聚类：为每个未聚类蛋白质寻找相似序列
        unclustered = set(proteins)
        while unclustered:
            seed = unclustered.pop()
            cluster = [seed]
            seed_seq = sequences[seed]
            for p in list(unclustered):
                p_seq = sequences[p]
                # 计算序列相似度（比对得分/最长序列长度）
                alignments = aligner.align(seed_seq, p_seq)
                max_score = max([a.score for a in alignments]) if alignments else 0
                similarity = max_score / max(len(seed_seq), len(p_seq))
                if similarity >= config.SEQUENCE_IDENTITY_THRESHOLD:
                    cluster.append(p)
                    unclustered.remove(p)
            if len(cluster) >= config.HYPEREDGE_NODE_THRESHOLD:
                node_indices = [self.protein2idx[p] for p in cluster]
                self.hyperedges.append(("sequence", set(node_indices)))
        print(f"构建序列相似性超边数量：{len([h for h in self.hyperedges if h[0] == 'sequence'])}")

    def construct_coexpression_hyperedges(self):
        """构建共表达超边（Pearson相关系数筛选）"""
        expr_data = self.data_loader.gene_exp_data.loc[self.data_loader.protein_ids]
        expr_matrix = expr_data.values
        num_samples = expr_matrix.shape[1]
        coexp_clusters = []
        
        # 计算蛋白质间共表达相关系数
        corr_matrix = np.corrcoef(expr_matrix)
        np.fill_diagonal(corr_matrix, 0)  # 排除自相关
        
        # 筛选高相关蛋白质组
        unclustered = set(range(self.num_proteins))
        while unclustered:
            seed = unclustered.pop()
            cluster = [seed]
            # 找到与seed高相关的蛋白质
            high_corr_indices = np.where(np.abs(corr_matrix[seed]) >= config.CO_EXPRESSION_CORR_THRESHOLD)[0]
            for idx in high_corr_indices:
                if idx in unclustered:
                    cluster.append(idx)
                    unclustered.remove(idx)
            if len(cluster) >= config.HYPEREDGE_NODE_THRESHOLD:
                self.hyperedges.append(("coexpression", set(cluster)))
        print(f"构建共表达超边数量：{len([h for h in self.hyperedges if h[0] == 'coexpression'])}")

    def calculate_hyperedge_weights(self):
        """基于信息增益计算超边初始权重"""
        labels = np.array([self.data_loader.essential_labels[p] for p in self.data_loader.protein_ids])
        weights = []
        
        for hyperedge_type, nodes in self.hyperedges:
            # 计算超边内关键蛋白质比例
            node_labels = labels[list(nodes)]
            essential_ratio = np.mean(node_labels)
            # 信息增益（简化计算：与全局比例的差异）
            global_essential_ratio = np.mean(labels)
            info_gain = abs(essential_ratio - global_essential_ratio)
            weights.append(info_gain)
        
        # 归一化权重
        weights = np.array(weights) / np.sum(weights)
        self.hyperedge_weights = weights
        print("超边权重计算完成")

    def run(self):
        """执行超图构建流程"""
        print("构建PPI超边...")
        self.construct_ppi_hyperedges()
        print("构建序列相似性超边...")
        self.construct_sequence_hyperedges()
        print("构建共表达超边...")
        self.construct_coexpression_hyperedges()
        print("计算超边初始权重...")
        self.calculate_hyperedge_weights()
        # 返回超边结构（节点集合列表）和权重
        hyperedge_sets = [nodes for _, nodes in self.hyperedges]
        return hyperedge_sets, self.hyperedge_weights