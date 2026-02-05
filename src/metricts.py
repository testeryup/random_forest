import pandas as pd
import numpy as np

def gini_index(groups, classes):
    n_instances = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = (group['Target'] == class_val).sum() / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

