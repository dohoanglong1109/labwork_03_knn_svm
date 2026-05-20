import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    KFold,
    LeaveOneOut,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, TruncatedSVD

OUTPUT_DIR = "../outputs/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_knn_baseline_vs_scaled(X_raw, X_scaled, y, optimal_k=5, rs=1):
    # Table 2.1
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=rs
    )
    X_train_scaled, X_test_scaled, _, _ = train_test_split(
        X_scaled, y, test_size=0.20, random_state=rs
    )

    # Raw Data
    knn_raw = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_raw.fit(X_train_raw, y_train)
    raw_error = (1 - accuracy_score(y_test, knn_raw.predict(X_test_raw))) * 100

    # Scaled Data
    knn_scaled = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_scaled.fit(X_train_scaled, y_train)
    scaled_error = (1 - accuracy_score(y_test, knn_scaled.predict(X_test_scaled))) * 100

    print(
        f"  -> Raw Error: {raw_error:.2f}% | Scaled Error: {scaled_error:.2f}% (k={optimal_k})"
    )
    return raw_error, scaled_error


def plot_knn_tuning_curve(X_scaled, y, rs=1):
    # Figure 2.1
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=rs
    )
    k_values = range(1, 31)
    train_errors, test_errors = [], []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_errors.append(1 - accuracy_score(y_train, knn.predict(X_train)))
        test_errors.append(1 - accuracy_score(y_test, knn.predict(X_test)))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_errors, marker="o", label="Train Error", color="#1f77b4")
    plt.plot(k_values, test_errors, marker="s", label="Test Error", color="#d62728")
    plt.xlabel("k value (Number of Neighbors)")
    plt.ylabel("Classification Error")
    plt.xticks(k_values)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.suptitle(
        "Figure 2.1: Training and Testing Error vs. k", y=0.02, fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(OUTPUT_DIR, "fig_2_1_tuning.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved Tuning Curve: {save_path}")


def plot_pca_components_tuning(X_scaled, y, optimal_k=7, rs=1):
    # Figure 3.1
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=rs
    )
    components_range = range(1, X_scaled.shape[1] + 1)
    errors = []

    for n in components_range:
        pca = PCA(n_components=n, random_state=rs)
        X_tr_pca = pca.fit_transform(X_train)
        X_te_pca = pca.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=optimal_k)
        knn.fit(X_tr_pca, y_train)
        errors.append(1 - accuracy_score(y_test, knn.predict(X_te_pca)))

    plt.figure(figsize=(10, 6))
    plt.plot(components_range, errors, marker="o", color="#2ca02c", linewidth=2)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Classification Error of k-NN")
    plt.xticks(components_range)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("Figure 3.1: k-NN Error vs. PCA Components", y=0.02, fontweight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(OUTPUT_DIR, "fig_3_1_pca_comps.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved PCA Components Curve: {save_path}")


def compare_pca_vs_svd(X_scaled, y, n_comps=8, optimal_k=7, rs=1):
    # Table 3.1
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=rs
    )

    # PCA
    pca = PCA(n_components=n_comps, random_state=rs)
    knn_pca = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_pca.fit(pca.fit_transform(X_train), y_train)
    pca_err = (1 - accuracy_score(y_test, knn_pca.predict(pca.transform(X_test)))) * 100

    # SVD
    svd = TruncatedSVD(n_components=n_comps, random_state=rs)
    knn_svd = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_svd.fit(svd.fit_transform(X_train), y_train)
    svd_err = (1 - accuracy_score(y_test, knn_svd.predict(svd.transform(X_test)))) * 100

    print(
        f"  -> PCA Error: {pca_err:.2f}% | SVD Error: {svd_err:.2f}% (Comps={n_comps})"
    )


def compare_5fold_vs_loo(X_scaled, y, optimal_k=19):
    """Table 4.1: Compares execution time and error of 5-Fold vs LOO."""
    knn = KNeighborsClassifier(n_neighbors=optimal_k)

    # 5-Fold
    start_5f = time.perf_counter()
    cv_5 = KFold(n_splits=5, shuffle=True, random_state=1)
    err_5f = (
        1 - cross_val_score(knn, X_scaled, y, cv=cv_5, scoring="accuracy").mean()
    ) * 100
    time_5f = time.perf_counter() - start_5f

    # LOO
    start_loo = time.perf_counter()
    cv_loo = LeaveOneOut()
    err_loo = (
        1 - cross_val_score(knn, X_scaled, y, cv=cv_loo, scoring="accuracy").mean()
    ) * 100
    time_loo = time.perf_counter() - start_loo

    print(
        f"  -> 5-Fold: {err_5f:.2f}% ({time_5f:.4f}s) | LOO: {err_loo:.2f}% ({time_loo:.4f}s)"
    )
