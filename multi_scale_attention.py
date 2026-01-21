import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（局部尺度特征学习）"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)

    def forward(self, x):
        # x: [batch_size, num_nodes_in_hyperedge, d_model]
        batch_size = x.size(0)
        
        # 投影并分多头：[batch_size, num_nodes, num_heads, d_k]
        q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数：[batch_size, num_heads, num_nodes, num_nodes]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 多头融合：[batch_size, num_nodes, d_model]
        output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.w_o(output)
        return output

class HyperedgeAwareHierarchicalAttention(nn.Module):
    """超边感知分层注意力（全局尺度特征学习）"""
    def __init__(self, d_model):
        super().__init__()
        self.node2hyperedge_attn = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1, bias=False)
        )
        self.hyperedge2node_attn = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1, bias=False)
        )
        self.dropout = nn.Dropout(config.DROPOUT_RATE)

    def forward(self, node_embeds, hyperedge_embeds, hyperedge_node_mapping):
        # node_embeds: [num_nodes, d_model]
        # hyperedge_embeds: [num_hyperedges, d_model]
        # hyperedge_node_mapping: 列表，每个元素为超边包含的节点索引列表
        num_hyperedges = len(hyperedge_node_mapping)
        num_nodes = node_embeds.size(0)
        
        # 第一步：节点→超边注意力聚合
        new_hyperedge_embeds = []
        for h in range(num_hyperedges):
            node_indices = hyperedge_node_mapping[h]
            if not node_indices:
                new_hyperedge_embeds.append(torch.zeros_like(hyperedge_embeds[h]))
                continue
            h_nodes = node_embeds[node_indices]  # [num_nodes_in_h, d_model]
            # 计算节点对超边的注意力权重
            h_embed = hyperedge_embeds[h].unsqueeze(0).repeat(len(node_indices), 1)  # [num_nodes_in_h, d_model]
            attn_input = torch.cat([h_nodes, h_embed], dim=-1)  # [num_nodes_in_h, 2*d_model]
            attn_scores = self.node2hyperedge_attn(attn_input)  # [num_nodes_in_h, 1]
            attn_weights = F.softmax(attn_scores, dim=0)  # [num_nodes_in_h, 1]
            attn_weights = self.dropout(attn_weights)
            # 超边嵌入更新
            new_h_embed = torch.sum(h_nodes * attn_weights, dim=0)  # [d_model]
            new_hyperedge_embeds.append(new_h_embed)
        new_hyperedge_embeds = torch.stack(new_hyperedge_embeds, dim=0)  # [num_hyperedges, d_model]
        
        # 第二步：超边→节点注意力聚合
        new_node_embeds = []
        for n in range(num_nodes):
            # 找到包含该节点的所有超边
            h_indices = [h for h, nodes in enumerate(hyperedge_node_mapping) if n in nodes]
            if not h_indices:
                new_node_embeds.append(node_embeds[n])
                continue
            n_hyperedges = new_hyperedge_embeds[h_indices]  # [num_hyperedges_for_n, d_model]
            # 计算超边对节点的注意力权重
            n_embed = node_embeds[n].unsqueeze(0).repeat(len(h_indices), 1)  # [num_hyperedges_for_n, d_model]
            attn_input = torch.cat([n_hyperedges, n_embed], dim=-1)  # [num_hyperedges_for_n, 2*d_model]
            attn_scores = self.hyperedge2node_attn(attn_input)  # [num_hyperedges_for_n, 1]
            attn_weights = F.softmax(attn_scores, dim=0)  # [num_hyperedges_for_n, 1]
            attn_weights = self.dropout(attn_weights)
            # 节点嵌入更新
            new_n_embed = torch.sum(n_hyperedges * attn_weights, dim=0)  # [d_model]
            new_node_embeds.append(new_n_embed)
        new_node_embeds = torch.stack(new_node_embeds, dim=0)  # [num_nodes, d_model]
        
        return new_node_embeds

class ParallelMultiScaleAttention(nn.Module):
    """并行多尺度注意力网络（融合局部+全局特征）"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.local_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.global_attn = HyperedgeAwareHierarchicalAttention(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)

    def forward(self, node_embeds, hyperedge_embeds, hyperedge_node_mapping):
        # 局部特征学习（每个超边内的节点交互）
        local_embeds = []
        for h_nodes in hyperedge_node_mapping:
            if len(h_nodes) == 0:
                local_embeds.extend([torch.zeros_like(node_embeds[0])]*len(h_nodes))
                continue
            h_node_embeds = node_embeds[list(h_nodes)].unsqueeze(0)  # [1, num_nodes_in_h, d_model]
            h_local_embed = self.local_attn(h_node_embeds).squeeze(0)  # [num_nodes_in_h, d_model]
            local_embeds.extend([h_local_embed[i] for i in range(len(h_nodes))])
        local_embeds = torch.stack(local_embeds, dim=0)  # [num_nodes, d_model]
        
        # 全局特征学习（超边-节点交互）
        global_embeds = self.global_attn(node_embeds, hyperedge_embeds, hyperedge_node_mapping)  # [num_nodes, d_model]
        
        # 哈达玛积融合 + 残差连接
        fused_embeds = self.layer_norm(node_embeds + self.dropout(local_embeds * global_embeds))
        return fused_embeds