import numpy as np


def accuracy_score(y_true, y_pred):
    """Tính accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Tính confusion matrix.
    Returns: np.ndarray shape (n_classes, n_classes)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def precision_score(y_true, y_pred, pos_label=1):
    """Precision = TP / (TP + FP)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, pos_label=1):
    """Recall = TP / (TP + FN)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    """F1 = 2 * precision * recall / (precision + recall)."""
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def classification_report(y_true, y_pred, labels=None, target_names=None):
    """In classification report đầy đủ."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    if target_names is None:
        target_names = [str(l) for l in labels]

    header = f"{'':>15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    lines = [header, "-" * len(header)]

    for label, name in zip(labels, target_names):
        p = precision_score(y_true, y_pred, pos_label=label)
        r = recall_score(y_true, y_pred, pos_label=label)
        f = f1_score(y_true, y_pred, pos_label=label)
        s = np.sum(y_true == label)
        lines.append(f"{name:>15} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10}")

    acc = accuracy_score(y_true, y_pred)
    total = len(y_true)
    lines.append("-" * len(header))
    lines.append(f"{'Accuracy':>15} {'':>10} {'':>10} {acc:>10.4f} {total:>10}")

    return "\n".join(lines)

