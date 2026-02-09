import numpy as np
import pandas as pd


class DataProcessor:
    """
    Xử lý dữ liệu: đọc CSV, xác định cột categorical/numerical,
    tách X/y, train-test split.
    """

    def __init__(self):
        self.feature_names = []
        self.cat_indices = []   # index của các cột categorical
        self.num_indices = []   # index của các cột numerical

    # ------------------------------------------------------------------
    def fit_transform(self, df, target_col, drop_cols=None):
        """
        Nhận DataFrame, trả về X (numpy object array), y (numpy array)
        và lưu lại thông tin feature types.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str
        drop_cols : list[str] or None
            Các cột cần bỏ (vd: customer_id).

        Returns
        -------
        X : np.ndarray (object dtype, shape [n_samples, n_features])
        y : np.ndarray
        """
        if drop_cols:
            df = df.drop(columns=drop_cols)

        self.feature_names = [c for c in df.columns if c != target_col]
        X_df = df[self.feature_names]
        y = df[target_col].values

        # Xác định categorical / numerical
        self.cat_indices = []
        self.num_indices = []
        for i, col in enumerate(self.feature_names):
            dtype = X_df[col].dtype
            if (dtype == 'object'
                    or dtype.name == 'category'
                    or pd.api.types.is_string_dtype(dtype)):
                self.cat_indices.append(i)
            else:
                self.num_indices.append(i)

        X = X_df.values  # object dtype do mixed types

        return X, y

    # ------------------------------------------------------------------
    def train_test_split(self, X, y, test_size=0.2, random_state=None):
        """Chia train/test theo tỷ lệ, có shuffle."""
        if random_state is not None:
            np.random.seed(random_state)

        n = len(X)
        indices = np.random.permutation(n)
        split = int(n * (1 - test_size))

        train_idx = indices[:split]
        test_idx = indices[split:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    # ------------------------------------------------------------------
    def summary(self):
        """In tóm tắt thông tin features."""
        print(f"Total features: {len(self.feature_names)}")
        print(f"Numerical ({len(self.num_indices)}): "
              f"{[self.feature_names[i] for i in self.num_indices]}")
        print(f"Categorical ({len(self.cat_indices)}): "
              f"{[self.feature_names[i] for i in self.cat_indices]}")
