from sklearn.metrics import accuracy_score, f1_score

def compare_labels_and_node_eval(labels, node_eval,mode='sf'):
    """
    比较labels和node_eval的变化，计算准确度和F1-score。
    labels: dict[int, int]，值为1或0
    node_eval: dict[int, str]，值为'准确'或'不准确'
    """
    if mode=='sf':
        # 只比较两者都存在的id
        common_ids = set(labels.keys()) & set(node_eval.keys())
        y_true = []
        y_pred = []
        for id_ in common_ids:
            # labels为1对应'准确'，labels为0对应'不准确'
            y_true.append(labels[id_])
            y_pred.append(1 if node_eval[id_] == 1 else 0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return {'accuracy': acc, 'f1_score': f1}
    else:
        # 只比较两者都存在的id
        common_ids = set(labels.keys()) & set(node_eval.keys())
        y_true = []
        y_pred = []
        for id_ in common_ids:
            # labels为1对应'准确'，labels为0对应'不准确'
            y_true.append(labels[id_])
            y_pred.append(1 if node_eval[id_] == '准确' else 0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return {'accuracy': acc, 'f1_score': f1}

def check_mrf_performance(labels, node_eval,gt):
    """
    检查MRF模型的性能，计算准确度和F1-score。
    labels: dict[int, int]，值为1或0
    node_eval: dict[int, int]，值为1或0
    """
    common_ids = set(labels.keys()) & set(node_eval.keys()) & set(gt.keys())
    y_true = []
    y_pred_labels = []
    y_pred_node_eval = []
    for id_ in common_ids:
        y_true.append(gt[id_])
        y_pred_labels.append(labels[id_])
        y_pred_node_eval.append(node_eval[id_])
    acc_labels = accuracy_score(y_true, y_pred_labels)
    f1_labels = f1_score(y_true, y_pred_labels)
    acc_node_eval = accuracy_score(y_true, y_pred_node_eval)
    f1_node_eval = f1_score(y_true, y_pred_node_eval)
    return {
        'labels': {'accuracy': acc_labels, 'f1_score': f1_labels},
        'node_eval': {'accuracy': acc_node_eval, 'f1_score': f1_node_eval}
    }


# 示例用法
# result = compare_labels_and_node_eval(labels, node_eval)
# print(result)