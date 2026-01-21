import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class FocalLoss(nn.Module):
    """焦点损失（缓解类别不平衡）"""
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [num_nodes, 1]（预测分数）
        # targets: [num_nodes, 1]（0/1标签）
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # 预测概率
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets.long()]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class NodeImportanceRegression(nn.Module):
    """结构感知双路径节点重要性回归"""
    def __init__(self, d_model, num_hyperedges):
        super().__init__()
        self.mlp_path = nn.Sequential(
            nn.Linear(d_model, config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.MLP_HIDDEN_DIM // 2, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.focal_loss = FocalLoss(gamma=config.FOCAL_LOSS_GAMMA)
        self.num_hyperedges = num_hyperedges

    def calculate_topology_contribution(self, hyperedge_node_mapping):
        """计算拓扑贡献因子（节点在超边中的注意力权重均值）"""
        num_nodes = max([max(nodes) for nodes in hyperedge_node_mapping]) + 1 if hyperedge_node_mapping else 0
        node_hyperedge_count = torch.zeros(num_nodes, dtype=torch.float32).to(config.DEVICE)
        topology_contribution = torch.zeros(num_nodes, 1).to(config.DEVICE)
        
        for h_nodes in hyperedge_node_mapping:
            h_size = len(h_nodes)
            if h_size == 0:
                continue
            # 简单计算：超边大小的倒数作为权重（可替换为注意力权重）
            h_weight = 1.0 / h_size
            for node in h_nodes:
                topology_contribution[node] += h_weight
                node_hyperedge_count[node] += 1
        
        # 归一化
        mask = node_hyperedge_count > 0
        topology_contribution[mask] /= node_hyperedge_count[mask].unsqueeze(1)
        return topology_contribution

    def forward(self, node_embeds, hyperedge_node_mapping, targets, train=True):
        """
        输入：节点嵌入、超边节点映射、标签
        输出：预测分数、焦点损失
        """
        # 路径1：MLP语义特征预测
        mlp_score = self.mlp_path(node_embeds)  # [num_nodes, 1]
        
        # 路径2：拓扑贡献因子
        topology_contrib = self.calculate_topology_contribution(hyperedge_node_mapping)  # [num_nodes, 1]
        
        # 门控融合
        gate_weight = self.gate(node_embeds)  # [num_nodes, 1]
        final_score = gate_weight * mlp_score + (1 - gate_weight) * topology_contrib  # [num_nodes, 1]
        
        # 计算损失（训练时）
        loss = self.focal_loss(final_score, targets.unsqueeze(1)) if train else None
        
        return final_score.squeeze(1), loss