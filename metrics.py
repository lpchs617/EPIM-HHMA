import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score

def calculate_metrics(predictions, targets, threshold=0.5):
    """计算多评估指标（SN、SP、PPV、NPV、F1、Accuracy）"""
    # 阈值划分预测标签
    pred_labels = (predictions >= threshold).astype(int)
    targets = targets.astype(int)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(targets, pred_labels).ravel()
    
    # 计算指标
    sn = tp / (tp + fn) if (tp + fn) != 0 else 0.0  # 灵敏度
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0  # 特异度
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0.0  # 阳性预测值
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0.0  # 阴性预测值
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0  # 准确率
    f1 = f1_score(targets, pred_labels) if (tp + fp + fn) != 0 else 0.0  # F1分数
    
    return {
        "SN": round(sn, 4),
        "SP": round(sp, 4),
        "PPV": round(ppv, 4),
        "NPV": round(npv, 4),
        "F1": round(f1, 4),
        "Accuracy": round(accuracy, 4)
    }

def get_optimal_threshold(predictions, targets):
    """基于PR曲线找到最优阈值（F1最大）"""
    precision, recall, thresholds = precision_recall_curve(targets, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return optimal_threshold