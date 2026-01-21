import torch
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
from config import config
from model.multi_scale_attention import ParallelMultiScaleAttention
from model.hypergraph_reconstruction import HypergraphReconstruction
from model.node_importance_regression import NodeImportanceRegression
from train.metrics import calculate_metrics, get_optimal_threshold

class Trainer:
    def __init__(self, num_proteins, initial_hyperedge_mapping, hyperedge_weights):
        self.num_proteins = num_proteins
        self.initial_hyperedge_mapping = initial_hyperedge_mapping
        self.hyperedge_weights = hyperedge_weights
        
        # 初始化模型
        self.multi_scale_attn = ParallelMultiScaleAttention(
            d_model=config.EMBEDDING_DIM,
            num_heads=config.ATTENTION_HEADS
        ).to(config.DEVICE)
        self.hypergraph_recon = HypergraphReconstruction(
            num_proteins=num_proteins,
            hyperedge_weights=hyperedge_weights,
            hyperedge_node_threshold=config.HYPEREDGE_NODE_THRESHOLD
        ).to(config.DEVICE)
        self.node_regression = NodeImportanceRegression(
            d_model=config.EMBEDDING_DIM,
            num_hyperedges=len(initial_hyperedge_mapping)
        ).to(config.DEVICE)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.multi_scale_attn.parameters()) +
            list(self.hypergraph_recon.parameters()) +
            list(self.node_regression.parameters()),
            lr=config.LEARNING_RATE
        )
        # 余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.MAX_EPOCHS)

    def prepare_data(self, protein_features, labels):
        """准备训练数据（特征、标签转Tensor）"""
        # protein_features: [num_proteins, feature_dim] → 映射为嵌入维度
        self.node_embeds = torch.nn.Embedding.from_pretrained(
            torch.tensor(protein_features, dtype=torch.float32),
            freeze=False
        ).to(config.DEVICE)
        # 超边嵌入初始化（随机）
        self.hyperedge_embeds = torch.randn(
            len(self.initial_hyperedge_mapping), config.EMBEDDING_DIM,
            dtype=torch.float32, requires_grad=True
        ).to(config.DEVICE)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(config.DEVICE)
        
        # 构建PPI邻接矩阵（简化：基于初始超边）
        self.ppi_adj_matrix = torch.zeros((self.num_proteins, self.num_proteins), dtype=torch.bool).to(config.DEVICE)
        for h_nodes in self.initial_hyperedge_mapping:
            for i in h_nodes:
                for j in h_nodes:
                    if i != j:
                        self.ppi_adj_matrix[i, j] = True

    def train_epoch(self, hyperedge_mapping):
        """单轮训练"""
        self.multi_scale_attn.train()
        self.hypergraph_recon.train()
        self.node_regression.train()
        
        # 1. 多尺度注意力特征学习
        updated_node_embeds = self.multi_scale_attn(
            self.node_embeds.weight, self.hyperedge_embeds, hyperedge_mapping
        )
        
        # 2. 超图重构
        new_hyperedge_mapping, recon_loss = self.hypergraph_recon(
            updated_node_embeds, self.ppi_adj_matrix, self.gene_exp_matrix, self.labels
        )
        
        # 3. 节点重要性回归
        predictions, focal_loss = self.node_regression(
            updated_node_embeds, new_hyperedge_mapping, self.labels, train=True
        )
        
        # 4. 联合损失
        total_loss = config.LAMBDA * recon_loss + (1 - config.LAMBDA) * focal_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.multi_scale_attn.parameters()) +
            list(self.hypergraph_recon.parameters()) +
            list(self.node_regression.parameters()),
            max_norm=1.0
        )  # 梯度裁剪
        self.optimizer.step()
        
        return total_loss.item(), new_hyperedge_mapping, predictions

    def validate(self, hyperedge_mapping, val_indices):
        """验证"""
        self.multi_scale_attn.eval()
        self.hypergraph_recon.eval()
        self.node_regression.eval()
        
        with torch.no_grad():
            updated_node_embeds = self.multi_scale_attn(
                self.node_embeds.weight, self.hyperedge_embeds, hyperedge_mapping
            )
            _, recon_loss = self.hypergraph_recon(
                updated_node_embeds, self.ppi_adj_matrix, self.gene_exp_matrix, self.labels
            )
            predictions, focal_loss = self.node_regression(
                updated_node_embeds, hyperedge_mapping, self.labels, train=False
            )
        
        # 计算验证集指标
        val_preds = predictions[val_indices].cpu().numpy()
        val_targets = self.labels[val_indices].cpu().numpy()
        optimal_thresh = get_optimal_threshold(val_preds, val_targets)
        metrics = calculate_metrics(val_preds, val_targets, optimal_thresh)
        total_loss = config.LAMBDA * recon_loss.item() + (1 - config.LAMBDA) * focal_loss.item()
        
        return metrics, total_loss, optimal_thresh

    def run_kfold_cv(self, protein_features, labels, gene_exp_matrix):
        """五折交叉验证训练"""
        self.gene_exp_matrix = torch.tensor(gene_exp_matrix, dtype=torch.float32).to(config.DEVICE)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_indices, val_indices) in enumerate(kf.split(range(self.num_proteins))):
            print(f"\n===== 第{fold+1}折 =====")
            self.prepare_data(protein_features, labels)
            
            best_val_f1 = 0.0
            best_metrics = None
            early_stop_count = 0
            current_hyperedge_mapping = self.initial_hyperedge_mapping.copy()
            
            for epoch in range(config.MAX_EPOCHS):
                # 训练
                train_loss, new_hyperedge_mapping, train_preds = self.train_epoch(current_hyperedge_mapping)
                current_hyperedge_mapping = new_hyperedge_mapping  # 更新超边
                self.scheduler.step()
                
                # 验证
                val_metrics, val_loss, val_thresh = self.validate(current_hyperedge_mapping, val_indices)
                
                # 早停
                if val_metrics["F1"] > best_val_f1:
                    best_val_f1 = val_metrics["F1"]
                    best_metrics = val_metrics
                    early_stop_count = 0
                    # 保存最优模型
                    torch.save({
                        "multi_scale_attn": self.multi_scale_attn.state_dict(),
                        "hypergraph_recon": self.hypergraph_recon.state_dict(),
                        "node_regression": self.node_regression.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_thresh": val_thresh
                    }, f"./models/best_model_fold{fold+1}.pth")
                else:
                    early_stop_count += 1
                    if early_stop_count >= config.EARLY_STOP_PATIENCE:
                        print(f"早停触发（epoch {epoch+1}）")
                        break
                
                # 打印日志
                if (epoch+1) % 50 == 0:
                    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_metrics['F1']:.4f}")
            
            fold_results.append(best_metrics)
            print(f"第{fold+1}折最优指标：{best_metrics}")
        
        # 计算五折平均指标
        avg_metrics = {
            "SN": np.mean([r["SN"] for r in fold_results]),
            "SP": np.mean([r["SP"] for r in fold_results]),
            "PPV": np.mean([r["PPV"] for r in fold_results]),
            "NPV": np.mean([r["NPV"] for r in fold_results]),
            "F1": np.mean([r["F1"] for r in fold_results]),
            "Accuracy": np.mean([r["Accuracy"] for r in fold_results])
        }
        print(f"\n五折交叉验证平均指标：{avg_metrics}")
        return avg_metrics