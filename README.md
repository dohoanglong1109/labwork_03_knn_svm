# K-Nearest Neighbor and Support Vector Machine Classifiers

## Objective

The primary objective of this lab work is to systematically evaluate and compare the performance of K-Nearest Neighbors (k-NN) and Support Vector Machine (SVM) algorithms. Through rigorous empirical testing, this project demonstrates the critical role of data preprocessing—specifically feature scaling, dimensionality reduction, and robust cross-validation—in building effective distance-based and margin-based classifiers.

---

## Repository Structure

```text
labwork_03_knn_svm/
├── data/                      <-- Raw dataset storage only
│   ├── heart_disease.csv
│   └── wine_quality.csv
├── outputs/                   <-- Execution outputs (automatically generated)
│   ├── figures/               <-- Saved visualization plots (Fig 1.1, 2.1, etc.)
│   └── tables_log.txt         <-- Optional text logs for metric tables
├── src/                       <-- Source code directory (.py scripts)
│   ├── data_loader.py         <-- Module: Data loading and preprocessing
│   ├── eda_visuals.py         <-- Module: Boxplots and 2D PCA visualizations
│   ├── knn_experiments.py     <-- Module: Hyperparameter tuning, PCA/SVD, and LOO
│   ├── svm_experiments.py     <-- Module: Linear vs RBF Kernel SVM comparisons
│   └── main.py                <-- Main orchestrator and execution script
├── README.md                  <-- Project documentation and final report
└── requirements.txt           <-- Dependency list (pandas, scikit-learn, etc.)
```

---

## How to Run

To replicate the experimental results and automatically generate the analytical figures in the `outputs/` directory, execute the following commands in your terminal environment:

```bash
# 1. Navigate into the labwork directory
cd labwork_03_knn_svm

# 2. Install the required external packages
pip install -r requirements.txt

# 3. Run the main orchestration script
python src/main.py
