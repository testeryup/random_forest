import numpy as np
import pandas as pd
from src import (
    DataProcessor, DecisionTree, RandomForest,
    accuracy_score, confusion_matrix, classification_report,
)


if __name__ == "__main__":
    # ===== 1. Đọc và xử lý dữ liệu =====
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION")
    print("=" * 60)

    data = pd.read_csv('data/loan_dataset.csv')
    print(f"\nDataset shape: {data.shape}")
    print(f"Target distribution:\n{data['loan_status'].value_counts()}\n")

    processor = DataProcessor()
    X, y = processor.fit_transform(
        data,
        target_col='loan_status',
        drop_cols=['customer_id'],
    )
    processor.summary()

    X_train, X_test, y_train, y_test = processor.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ===== 2. Decision Tree =====
    print("\n" + "=" * 60)
    print("DECISION TREE")
    print("=" * 60)

    tree = DecisionTree(
        max_depth=10,
        min_samples_split=5,
        min_gain=0.01,
        cat_features=processor.cat_indices,
    )
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)

    acc_tree = accuracy_score(y_test, y_pred_tree)
    print(f"\nAccuracy: {acc_tree:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_tree)}")
    print(f"\n{classification_report(y_test, y_pred_tree, target_names=['Rejected(0)', 'Approved(1)'])}")

    # Hiển thị cây (chỉ vài nhánh đầu)
    # tree.print_tree(feature_names=processor.feature_names)

    # ===== 3. Random Forest =====
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    rf = RandomForest(
        n_trees=100,
        max_depth=20,
        min_samples_split=2,
        min_gain=0.0,       # RF trees nên grow fully, ensemble lo variance
        max_features='sqrt',
        cat_features=processor.cat_indices,
        random_state=42,
    )
    print("\nTraining Random Forest...")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\nAccuracy: {acc_rf:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
    print(f"\n{classification_report(y_test, y_pred_rf, target_names=['Rejected(0)', 'Approved(1)'])}")

    # OOB Score (built-in validation không cần test set riêng)
    oob = rf.oob_score()
    if oob is not None:
        print(f"\nOOB Score: {oob:.4f}")

    # Feature importance
    importances = rf.feature_importances(len(processor.feature_names))
    print("\nTop 10 Feature Importances:")
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank}. {processor.feature_names[idx]:30s} {importances[idx]:.4f}")

    # ===== 4. So sánh =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Decision Tree Accuracy: {acc_tree:.4f}")
    print(f"  Random Forest Accuracy: {acc_rf:.4f}")
    print(f"  Improvement:            {acc_rf - acc_tree:+.4f}")

    



