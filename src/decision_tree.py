import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, gini=None, n_samples=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gini = gini
        self.n_samples = n_samples

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_gain=0.01):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root = None
        self.feature_importances = {}

    def _gini(self, y):
        n = len(y)
        if n == 0: return 0
        # Dùng np.unique để đếm nếu nhãn là chuỗi hoặc số
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n
        return 1 - np.sum(probabilities**2)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        n_sample, n_features = X.shape
        parent_gini = self._gini(y)

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Phân biệt cách chia cho Số và Chữ
                if isinstance(thr, str):
                    left_indices = np.where(X_column == thr)[0]
                    right_indices = np.where(X_column != thr)[0]
                else:
                    left_indices = np.where(X_column <= thr)[0]
                    right_indices = np.where(X_column > thr)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                gini_left, gini_right = self._gini(y_left), self._gini(y_right)
                
                child_gini = (len(y_left) / n_sample) * gini_left + (len(y_right) / n_sample) * gini_right
                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
                    
        return split_idx, split_thresh, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_labels = np.unique(y)

        # Điều kiện dừng
        if (n_samples < self.min_samples_split or depth >= self.max_depth or len(unique_labels) == 1):
            return Node(value=self._most_common_label(y), n_samples=n_samples, gini=self._gini(y))

        feat_idx, threshold, gain = self._best_split(X, y)

        if gain < self.min_gain:
            return Node(value=self._most_common_label(y), n_samples=n_samples, gini=self._gini(y))

        # Lưu Feature Importance
        self.feature_importances[feat_idx] = self.feature_importances.get(feat_idx, 0) + (n_samples * gain)

        # Tách indices cho bước đệ quy
        if isinstance(threshold, str):
            left_idx = np.where(X[:, feat_idx] == threshold)[0]
            right_idx = np.where(X[:, feat_idx] != threshold)[0]
        else:
            left_idx = np.where(X[:, feat_idx] <= threshold)[0]
            right_idx = np.where(X[:, feat_idx] > threshold)[0]

        left = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(feat_idx, threshold, left, right, gini=self._gini(y), n_samples=n_samples)

    def _most_common_label(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def fit(self, X, y):
        # Chuyển đổi sang numpy array để tránh lỗi index của Pandas
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        # Logic rẽ nhánh khi Predict
        val = x[node.feature]
        if isinstance(node.threshold, str):
            if val == node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:
            if val <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def print_tree(self, node=None, depth=0):
        if node is None: node = self.root
        indent = "    " * depth
        if node.is_leaf():
            print(f"{indent}Leaf: {node.value} (gini={node.gini:.4f})")
        else:
            op = "==" if isinstance(node.threshold, str) else "<="
            print(f"{indent}[X{node.feature} {op} {node.threshold}] (gini={node.gini:.4f}, samples={node.n_samples})")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)