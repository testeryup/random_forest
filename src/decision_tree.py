import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_features=None, verbose=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.verbose = verbose
        self.root = None

    def _calulate_gini(self, y):
        n = len(y)
        if n==0: return 0
        counts = np.bincount(y)
        probabilities = counts / n
        return 1 - np.sum(probabilities**2)
    
    def _best_split(self, X, y, feat_indices):
        