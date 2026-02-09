import numpy as np
from src.node import Node


class DecisionTree:
    """
    Decision Tree classifier hỗ trợ cả dữ liệu số (numerical) và phân loại (categorical).

    Parameters
    ----------
    max_depth : int
        Độ sâu tối đa của cây.
    min_samples_split : int
        Số mẫu tối thiểu để tiếp tục chia.
    min_gain : float
        Information gain tối thiểu để chấp nhận một split.
    n_features : int or None
        Số feature ngẫu nhiên xét tại mỗi split (dùng cho Random Forest).
        None = xét tất cả features.
    cat_features : list[int] or None
        Danh sách index của các cột categorical.
    max_bins : int or None
        Giới hạn số candidate split cho numerical features.
        None = dùng tất cả. 256 là mặc định tốt (tương tự LightGBM).
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_gain=0.01,
                 n_features=None, cat_features=None, max_bins=256):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.n_features = n_features
        self.cat_features = set(cat_features) if cat_features else set()
        self.max_bins = max_bins
        self.root = None
        self.feature_importances = {}

    # ------------------------------------------------------------------
    # Gini impurity
    # ------------------------------------------------------------------
    def _gini(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return 1.0 - np.sum(probs ** 2)

    # ------------------------------------------------------------------
    # Kiểm tra feature có phải categorical không
    # ------------------------------------------------------------------
    def _is_categorical(self, feat_idx):
        return feat_idx in self.cat_features

    # ------------------------------------------------------------------
    # Tìm split tốt nhất (vectorized)
    # ------------------------------------------------------------------
    def _best_split(self, X, y, feature_indices):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        n_samples = len(y)
        parent_gini = self._gini(y)

        # Pre-compute class mapping 1 lần cho tất cả features
        classes, y_mapped = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        total_class_counts = np.bincount(y_mapped, minlength=n_classes)

        for feat_idx in feature_indices:
            col = X[:, feat_idx]

            if self._is_categorical(feat_idx):
                # --- Categorical: thử từng giá trị unique (ít categories) ---
                categories = np.unique(col)
                for cat in categories:
                    left_mask = col == cat
                    n_left = np.sum(left_mask)
                    n_right = n_samples - n_left
                    if n_left == 0 or n_right == 0:
                        continue

                    left_counts = np.bincount(y_mapped[left_mask], minlength=n_classes)
                    right_counts = total_class_counts - left_counts
                    gini_left = 1.0 - np.sum((left_counts / n_left) ** 2)
                    gini_right = 1.0 - np.sum((right_counts / n_right) ** 2)
                    gini_child = (n_left / n_samples) * gini_left \
                               + (n_right / n_samples) * gini_right
                    gain = parent_gini - gini_child

                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_thresh = cat
            else:
                # --- Numerical: vectorized sorted-sweep ---
                thresh, gain = self._best_numerical_split(
                    col, y_mapped, n_classes, n_samples, parent_gini)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh, best_gain

    # ------------------------------------------------------------------
    # Vectorized numerical split – O(n log n) thay vì O(n × k)
    # ------------------------------------------------------------------
    def _best_numerical_split(self, col, y_mapped, n_classes, n_samples, parent_gini):
        """
        Tìm threshold tốt nhất bằng sorted sweep + cumulative sum.
        Thay vì loop qua hàng ngàn thresholds, dùng numpy vectorized.
        """
        col_float = col.astype(np.float64)
        sorted_idx = np.argsort(col_float, kind='quicksort')
        sorted_vals = col_float[sorted_idx]
        sorted_y = y_mapped[sorted_idx]

        # Tìm vị trí giá trị thay đổi → split candidates
        change_mask = sorted_vals[:-1] != sorted_vals[1:]
        change_indices = np.where(change_mask)[0]

        if len(change_indices) == 0:
            return None, -1.0

        # Giới hạn số candidates (histogram-style, tương tự LightGBM)
        if self.max_bins and len(change_indices) > self.max_bins:
            step = len(change_indices) / self.max_bins
            sampled = np.unique(np.floor(
                np.arange(self.max_bins) * step).astype(int))
            change_indices = change_indices[sampled]

        # One-hot encode sorted labels → cumulative class counts
        one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), sorted_y] = 1.0
        cum_counts = np.cumsum(one_hot, axis=0)

        # Lấy class counts tại mỗi split candidate
        left_counts = cum_counts[change_indices]        # (n_splits, n_classes)
        total_counts = cum_counts[-1]                    # (n_classes,)
        right_counts = total_counts - left_counts

        n_left = (change_indices + 1).astype(np.float64).reshape(-1, 1)
        n_right = n_samples - n_left

        # Vectorized Gini tại tất cả candidates cùng lúc
        gini_left = 1.0 - np.sum((left_counts / n_left) ** 2, axis=1)
        gini_right = 1.0 - np.sum((right_counts / n_right) ** 2, axis=1)

        child_gini = (n_left.ravel() / n_samples) * gini_left \
                   + (n_right.ravel() / n_samples) * gini_right
        gains = parent_gini - child_gini

        best_pos = np.argmax(gains)
        best_gain = gains[best_pos]
        i = change_indices[best_pos]
        best_thresh = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0

        return best_thresh, best_gain

    # ------------------------------------------------------------------
    # Xây dựng cây đệ quy
    # ------------------------------------------------------------------
    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_total_features = X.shape[1]
        unique_labels = np.unique(y)

        # Điều kiện dừng
        if (n_samples < self.min_samples_split
                or depth >= self.max_depth
                or len(unique_labels) == 1):
            return Node(value=self._most_common_label(y),
                        n_samples=n_samples, gini=self._gini(y))

        # Chọn ngẫu nhiên feature subset (cho Random Forest)
        if self.n_features and self.n_features < n_total_features:
            feature_indices = np.random.choice(
                n_total_features, self.n_features, replace=False)
        else:
            feature_indices = np.arange(n_total_features)

        feat_idx, threshold, gain = self._best_split(X, y, feature_indices)

        if feat_idx is None or gain < self.min_gain:
            return Node(value=self._most_common_label(y),
                        n_samples=n_samples, gini=self._gini(y))

        # Lưu Feature Importance (weighted by n_samples)
        self.feature_importances[feat_idx] = \
            self.feature_importances.get(feat_idx, 0) + (n_samples * gain)

        # Tách dữ liệu
        if self._is_categorical(feat_idx):
            left_mask = X[:, feat_idx] == threshold
        else:
            left_mask = X[:, feat_idx].astype(float) <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feat_idx, threshold, left, right,
                    gini=self._gini(y), n_samples=n_samples)

    # ------------------------------------------------------------------
    def _most_common_label(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.feature_importances = {}
        self.root = self._build_tree(X, y)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if self._is_categorical(node.feature):
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:
            if float(x[node.feature]) <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # ------------------------------------------------------------------
    # In cây
    # ------------------------------------------------------------------
    def print_tree(self, node=None, depth=0, feature_names=None):
        if node is None:
            node = self.root
        indent = "    " * depth

        if node.is_leaf():
            print(f"{indent}Leaf: {node.value} (gini={node.gini:.4f}, samples={node.n_samples})")
        else:
            fname = feature_names[node.feature] if feature_names else f"X{node.feature}"
            op = "==" if self._is_categorical(node.feature) else "<="
            print(f"{indent}[{fname} {op} {node.threshold}] "
                  f"(gini={node.gini:.4f}, samples={node.n_samples})")
            self.print_tree(node.left, depth + 1, feature_names)
            self.print_tree(node.right, depth + 1, feature_names)