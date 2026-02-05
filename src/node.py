class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, gini=None, n_samples=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value       # lá
        self.gini = gini
        self.n_samples = n_samples