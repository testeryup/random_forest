import numpy as np
from src.decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest classifier (Bagging + Random Feature Subset).

    Parameters
    ----------
    n_trees : int
        Số lượng cây quyết định.
    max_depth : int
        Độ sâu tối đa mỗi cây.
    min_samples_split : int
        Số mẫu tối thiểu để split.
    min_gain : float
        Information gain tối thiểu.
    max_features : str, int, or float
        Số features ngẫu nhiên tại mỗi split:
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - int:   số cụ thể
        - float: tỷ lệ (0‒1)
        - None:  dùng tất cả features
    cat_features : list[int] or None
        Index các cột categorical.
    random_state : int or None
        Seed cho reproducibility.
    """

    def __init__(self, n_trees=100, max_depth=10, min_samples_split=5,
                 min_gain=0.01, max_features='sqrt', cat_features=None,
                 random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_features = max_features
        self.cat_features = cat_features
        self.random_state = random_state
        self.trees = []
        self._oob_indices = []   # lưu OOB indices cho từng cây
        self._X_train = None
        self._y_train = None

    # ------------------------------------------------------------------
    def _get_n_features(self, n_total):
        """Tính số features random subset từ max_features."""
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_total)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_total)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total))
        return n_total  # None → dùng tất cả

    # ------------------------------------------------------------------
    def _bootstrap_sample(self, X, y):
        """Tạo bootstrap sample (sampling with replacement).
        Trả về thêm OOB indices (samples không nằm trong bootstrap)."""
        n_samples = X.shape[0]
        bag_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[bag_indices] = False
        oob_indices = np.where(oob_mask)[0]
        return X[bag_indices], y[bag_indices], oob_indices

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Huấn luyện Random Forest."""
        X = np.array(X)
        y = np.array(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_features = self._get_n_features(X.shape[1])
        self.trees = []
        self._oob_indices = []
        self._X_train = X
        self._y_train = y

        for i in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain,
                n_features=n_features,
                cat_features=self.cat_features,
            )
            X_boot, y_boot, oob_idx = self._bootstrap_sample(X, y)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self._oob_indices.append(oob_idx)

            if (i + 1) % 10 == 0 or (i + 1) == self.n_trees:
                print(f"  Trained {i + 1}/{self.n_trees} trees")

    # ------------------------------------------------------------------
    def predict(self, X):
        """Dự đoán bằng majority vote."""
        X = np.array(X)
        # Mỗi hàng = 1 cây, mỗi cột = 1 sample
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        y_pred = []
        for j in range(X.shape[0]):
            vals, counts = np.unique(all_preds[:, j], return_counts=True)
            y_pred.append(vals[np.argmax(counts)])
        return np.array(y_pred)

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """Trả về xác suất thuộc class 1 (dùng cho loan_status)."""
        X = np.array(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # Tính tỷ lệ cây vote class 1
        proba = np.mean(all_preds.astype(float) == 1, axis=0)
        return proba

    # ------------------------------------------------------------------
    def oob_score(self):
        """
        Out-of-Bag score: mỗi sample được predict bởi các cây
        KHÔNG chứa nó trong bootstrap → validation miễn phí.
        """
        if self._X_train is None or len(self.trees) == 0:
            return None

        n_samples = len(self._y_train)
        # Tích lũy votes cho mỗi sample (chỉ từ cây OOB)
        vote_counts = {}  # sample_idx → {label: count}

        for tree, oob_idx in zip(self.trees, self._oob_indices):
            if len(oob_idx) == 0:
                continue
            preds = tree.predict(self._X_train[oob_idx])
            for idx, pred in zip(oob_idx, preds):
                if idx not in vote_counts:
                    vote_counts[idx] = {}
                vote_counts[idx][pred] = vote_counts[idx].get(pred, 0) + 1

        # Majority vote cho mỗi sample
        correct = 0
        total = 0
        for idx, votes in vote_counts.items():
            best_label = max(votes, key=votes.get)
            if best_label == self._y_train[idx]:
                correct += 1
            total += 1

        if total == 0:
            return None
        return correct / total

    # ------------------------------------------------------------------
    def feature_importances(self, n_features):
        """Tổng hợp feature importance từ tất cả các cây."""
        total = np.zeros(n_features)
        for tree in self.trees:
            for feat_idx, imp in tree.feature_importances.items():
                total[feat_idx] += imp
        # Normalize
        s = total.sum()
        if s > 0:
            total /= s
        return total
