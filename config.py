import torch

class Config:
    # 数据配置
    PPI_DATA_PATH = "./data/dip_ppi.txt"  # PPI网络数据路径
    GENE_EXP_PATH = "./data/gse3431_expr.csv"  # 基因表达数据路径
    SEQUENCE_PATH = "./data/yeast_sequences.fasta"  # 氨基酸序列路径
    ESSENTIAL_LABEL_PATH = "./data/essential_proteins.txt"  # 关键蛋白质标签路径
    
    # 超图配置
    MCL_INFLATION = 2.0  # MCL聚类膨胀系数
    SEQUENCE_IDENTITY_THRESHOLD = 0.5  # 序列相似性阈值
    CO_EXPRESSION_CORR_THRESHOLD = 0.8  # 共表达相关系数阈值
    HYPEREDGE_NODE_THRESHOLD = 5  # 超边最小节点数
    
    # 模型配置
    EMBEDDING_DIM = 256  # 特征嵌入维度
    ATTENTION_HEADS = 8  # 多头注意力头数
    MLP_HIDDEN_DIM = 512  # MLP隐藏层维度
    DROPOUT_RATE = 0.1  # Dropout概率
    
    # 训练配置
    LEARNING_RATE = 0.01  # 最优学习率（实验验证）
    BATCH_SIZE = 128
    MAX_EPOCHS = 500
    EARLY_STOP_PATIENCE = 20  # 早停耐心值
    LAMBDA = 0.5  # 损失权重系数（重构损失:焦点损失=1:1）
    FOCAL_LOSS_GAMMA = 2.0  # 焦点损失gamma参数
    
    # 硬件配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # 数据加载线程数

config = Config()