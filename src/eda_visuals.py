import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ensure the output directory exists before saving files
OUTPUT_DIR = "../outputs/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_boxplot(X_raw):
    # Figure 1.1
    plt.figure(figsize=(14, 6))

    # Draw boxplot using raw data
    sns.boxplot(data=X_raw, palette="Set2")

    # Configure academic plot style
    plt.title("Scale Disparity Across All Features", fontsize=14, fontweight="bold")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Value Range", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save image and close figure to free memory
    save_path = os.path.join(OUTPUT_DIR, "fig_1_1_boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"-> Successfully saved: {save_path}")


def plot_pca_2d(X_heart_scaled, y_heart, X_wine_scaled, y_wine):
    # Figure 1.2
    # Helper function to compute PCA
    def get_pca_dataframe(X_scaled, y):
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(data=components, columns=["PC1", "PC2"])
        df_pca["target"] = y.values
        return df_pca

    # Apply PCA to both scaled datasets
    pca_heart = get_pca_dataframe(X_heart_scaled, y_heart)
    pca_wine = get_pca_dataframe(X_wine_scaled, y_wine)

    # Initialize a 1x2 subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Heart Disease
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="target",
        data=pca_heart,
        palette="Set1",
        alpha=0.8,
        ax=axes[0],
    )
    axes[0].set_title("Heart Disease PCA (2D)", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right plot: Wine Quality
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="target",
        data=pca_wine,
        palette="viridis",
        alpha=0.7,
        ax=axes[1],
    )
    axes[1].set_title("Wine Quality PCA (2D)", fontsize=13, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Add figure caption
    plt.suptitle(
        "Figure 1.2: 2D PCA projection of Heart Disease (left) and Wine Quality (right).",
        y=0.02,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save image and close figure
    save_path = os.path.join(OUTPUT_DIR, "fig_1_2_pca.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"-> Successfully saved: {save_path}")
