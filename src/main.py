from data_loader import load_heart_data, load_wine_data
from eda_visuals import plot_boxplot, plot_pca_2d
from knn_experiments import (
    run_knn_baseline_vs_scaled,
    plot_knn_tuning_curve,
    plot_pca_components_tuning,
    compare_pca_vs_svd,
    compare_5fold_vs_loo,
)
from svm_experiments import compare_svm_kernels


def main():
    print("=== PIPELINE STARTED ===")

    # 1. Load Data
    print("\n[1/4] Loading and cleaning datasets...")
    X_heart, X_heart_scaled, y_heart = load_heart_data("../data/heart_disease.csv")
    X_wine, X_wine_scaled, y_wine = load_wine_data("../data/wine_quality-red.csv")

    # 2. EDA Visuals
    print("\n[2/4] Generating EDA Visuals...")
    plot_boxplot(X_heart)
    plot_pca_2d(X_heart_scaled, y_heart, X_wine_scaled, y_wine)

    # 3. k-NN Experiments
    print("\n[3/4] Running k-NN Experiments...")
    print("--- Heart Disease Baseline (k=5) ---")
    run_knn_baseline_vs_scaled(X_heart, X_heart_scaled, y_heart, optimal_k=5)
    plot_knn_tuning_curve(X_heart_scaled, y_heart)

    print("--- Wine Quality PCA/SVD (k=7) ---")
    plot_pca_components_tuning(X_wine_scaled, y_wine, optimal_k=7)
    compare_pca_vs_svd(X_wine_scaled, y_wine, n_comps=8, optimal_k=7)

    print("--- Cross-Validation vs LOO (Heart Disease, k=19) ---")
    compare_5fold_vs_loo(X_heart_scaled, y_heart, optimal_k=19)

    # 4. SVM Experiments
    print("\n[4/4] Running SVM Experiments...")
    print("--- Heart Disease SVM ---")
    compare_svm_kernels(X_heart_scaled, y_heart)
    print("--- Wine Quality SVM ---")
    compare_svm_kernels(X_wine_scaled, y_wine)

    print("\n=== PIPELINE FINISHED. Check 'outputs/' folder for figures! ===")


if __name__ == "__main__":
    main()
