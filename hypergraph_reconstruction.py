import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import jaccard
from dtw import dtw  # 第三方DTW库（需安装：pip install dtw-python）
from config import config

class MultiModalBiologicalSemanticDistance(nn.Module):
    """多模态生物语义距离计算"""
    def __init__(self, num_proteins, hyperedge_weights):
        super().__init__()
        self.num_proteins = num_proteins
        self.hyperedge_weights = torch.tensor(hyperedge_weights, dtype=torch.float32).to(config.DEVICE)
        self.penalty_coef = torch.tensor(1.0, dtype=torch.float32).to(config.DEVICE)  # 惩罚系数

    def calculate_ppi_distance(self, ppi_adj_matrix):
        """计算PPI拓扑距离（加权Jaccard改进型）"""
        ppi_dist = torch.zeros((self.num_proteins, self.num_proteins)).to(config.DEVICE)
        for i in range(self.num_proteins):
            for j in range(i+1, self.num_proteins):
                # 加权Jaccard距离：1 - (交集权重和)/(并集权重和)
                intersection = torch.logical_and(ppi_adj_matrix[i], ppi_adj_matrix[j]).float().sum()
                union = torch.logical_or(ppi_adj_matrix[i], ppi_adj_matrix[j]).float().sum()
                ppi_dist[i, j] = 1 - (intersection / union) if union != 0 else 1.0
                ppi_dist[j, i] = ppi_dist[i, j]
        return ppi_dist

    def calculate_sequence_distance(self, sequence_embeds):
        """计算序列相似性距离（1 - 余弦相似度）"""
        seq_sim = torch.matmul(sequence_embeds, sequence_embeds.t()) / (
            torch.norm(sequence_embeds, dim=1, keepdim=True) * torch.norm(sequence_embeds, dim=1, keepdim=True).t()
        )
        seq_dist = 1 - seq_sim
        return seq_dist

    def calculate_coexpression_distance(self, gene_exp_matrix):
        """计算基因表达距离（DTW距离）"""
        exp_dist = torch.zeros((self.num_proteins, self.num_proteins)).to(config.DEVICE)
        for i in range(self.num_proteins):
            for j in range(i+1, self.num_proteins):
                dist, _, _, _ = dtw(gene_exp_matrix[i].cpu().numpy(), gene_exp_matrix[j].cpu().numpy())
                exp_dist[i, j] = dist / max(len(gene_exp_matrix[i]), len(gene_exp_matrix[j]))  # 归一化
                exp_dist[j, i] = exp_dist[i, j]
        return exp_dist

    def forward(self, ppi_adj_matrix, sequence_embeds, gene_exp_matrix, essential_labels):
        """计算多模态生物语义距离（MBSD）"""
        # 计算单模态距离
        ppi_dist = self.calculate_ppi_distance(ppi_adj_matrix)
        seq_dist = self.calculate_sequence_distance(sequence_embeds)
        exp_dist = self.calculate_coexpression_distance(gene_exp_matrix)
        
        # 功能一致性惩罚项（同一超边内标签不一致惩罚）
        func_penalty = torch.zeros((self.num_proteins, self.num_proteins)).to(config.DEVICE)
        for i in range(self.num_proteins):
            for j in range(self.num_proteins):
                if essential_labels[i] != essential_labels[j]:
                    func_penalty[i, j] = 1.0
        
        # MBSD公式：加权求和 + 惩罚项
        mbsd = (
            self.hyperedge_weights[0] * ppi_dist +
            self.hyperedge_weights[1] * seq_dist +
            self.hyperedge_weights[2] * exp_dist +
            self.penalty_coef * func_penalty
        )
        return mbsd

class HypergraphReconstruction(nn.Module):
    """基于MBSD的超图重构"""
    def __init__(self, num_proteins, hyperedge_weights, hyperedge_node_threshold):
        super().__init__()
        self.num_proteins = num_proteins
        self.hyperedge_node_threshold = hyperedge_node_threshold
        self.mbsd_calculator = MultiModalBiologicalSemanticDistance(num_proteins, hyperedge_weights)

    def forward(self, node_embeds, ppi_adj_matrix, gene_exp_matrix, essential_labels):
        """
        输入：节点嵌入、PPI邻接矩阵、基因表达矩阵、关键蛋白质标签
        输出：重构后的超边节点映射、超图重构损失
        """
        # 计算MBSD矩阵
        mbsd_matrix = self.mbsd_calculator(ppi_adj_matrix, node_embeds, gene_exp_matrix, essential_labels)
        
        # 重构超边：为每个蛋白质筛选距离最小的k个节点
        new_hyperedge_mapping = []
        for i in range(self.num_proteins):
            # 筛选距离最小的前k个节点（k=hyperedge_node_threshold）
            distances = mbsd_matrix[i].cpu().numpy()
            nearest_indices = np.argsort(distances)[:self.hyperedge_node_threshold+1]  # +1包含自身
            nearest_indices = [idx for idx in nearest_indices if idx != i]  # 去除自身
            if len(nearest_indices) >= self.hyperedge_node_threshold:
                new_hyperedge_mapping.append(set(nearest_indices))
        
        # 计算超图重构损失（超边与节点嵌入的距离损失）
        recon_loss = 0.0
        for h_nodes in new_hyperedge_mapping:
            if not h_nodes:
                continue
            h_node_embeds = node_embeds[list(h_nodes)]
            h_embed = h_node_embeds.mean(dim=0)  # 超边嵌入（节点平均）
            # 超边与节点的距离损失
            node_distances = torch.norm(h_node_embeds - h_embed.unsqueeze(0), dim=1)
            recon_loss += node_distances.mean()
        recon_loss /= len(new_hyperedge_mapping) if new_hyperedge_mapping else 1.0
        
        return new_hyperedge_mapping, recon_loss