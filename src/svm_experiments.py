from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def compare_svm_kernels(X_scaled, y, rs=1):
    # Table 5.1
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=rs
    )

    # Linear SVM
    svm_linear = SVC(kernel="linear", random_state=rs)
    svm_linear.fit(X_train, y_train)
    linear_err = (1 - accuracy_score(y_test, svm_linear.predict(X_test))) * 100

    # RBF SVM
    svm_rbf = SVC(kernel="rbf", random_state=rs)
    svm_rbf.fit(X_train, y_train)
    rbf_err = (1 - accuracy_score(y_test, svm_rbf.predict(X_test))) * 100

    print(f"  -> Linear Error: {linear_err:.2f}% | RBF Error: {rbf_err:.2f}%")
