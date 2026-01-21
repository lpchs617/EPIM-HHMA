import numpy as np
from data_processing.data_loader import DataLoader
from data_processing.hypergraph_construction import HypergraphConstructor
from train.trainer import Trainer
from config import config

def main():
    # 1. 数据加载
    print("===== 数据加载 =====")
    data_loader = DataLoader().run()
    proteins = data_loader.protein_ids
    num_proteins = len(proteins)
    
    # 2. 特征构建（拼接序列特征、基因表达特征）
    print("===== 特征构建 =====")
    # 序列特征：简单编码（氨基酸One-Hot，简化为20维）
    aa_dict = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9,
               'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
    sequence_features = []
    for p in proteins:
        seq = data_loader.sequence_data[p]
        # 序列长度统一为100（截断/填充）
        seq = seq[:100] if len(seq) >= 100 else seq.ljust(100, 'X')
        # One-Hot编码（100*20 → 2000维）
        one_hot = np.zeros((100, 20))
        for i, aa in enumerate(seq):
            if aa in aa_dict:
                one_hot[i, aa_dict[aa]] = 1.0
        sequence_features.append(one_hot.flatten())
    sequence_features = np.array(sequence_features)
    
    # 基因表达特征（36个时间点 → 36维）
    gene_exp_features = data_loader.gene_exp_data.loc[proteins].values
    
    # 拼接特征（2000 + 36 = 2036维）
    protein_features = np.hstack([sequence_features, gene_exp_features])
    # 映射为嵌入维度（2036 → 256）
    from sklearn.decomposition import PCA
    pca = PCA(n_components=config.EMBEDDING_DIM)
    protein_features = pca.fit_transform(protein_features)
    
    # 标签（0/1）
    labels = np.array([data_loader.essential_labels[p] for p in proteins])
    
    # 3. 异质超图构建
    print("===== 异质超图构建 =====")
    hypergraph_constructor = HypergraphConstructor(data_loader)
    initial_hyperedge_mapping, hyperedge_weights = hypergraph_constructor.run()
    
    # 4. 模型训练
    print("===== 模型训练 =====")
    trainer = Trainer(
        num_proteins=num_proteins,
        initial_hyperedge_mapping=initial_hyperedge_mapping,
        hyperedge_weights=hyperedge_weights
    )
    trainer.run_kfold_cv(protein_features, labels, gene_exp_features)
    
    print("\n===== 训练完成 =====")

if __name__ == "__main__":
    main()