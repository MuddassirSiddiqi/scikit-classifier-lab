import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Add Gaussian noise to the iris dataset
X_iris_noisy = X_iris + np.random.normal(0, 0.2, X_iris.shape)

# Define the classifiers
classifiers = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10),
    "Naive Bayes": GaussianNB(),
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "Kernel SVM": SVC(kernel="rbf", gamma=2, C=1)
}

# Evaluation function
def evaluate_classifiers(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    print(f"\n{dataset_name} - Test labels (y_test): {y_test}")
    for name, clf in classifiers.items():
        model = make_pipeline(StandardScaler(), clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{dataset_name} - {name} Accuracy: {acc:.4f}")

# Evaluate classifiers on both clean and noisy iris datasets
evaluate_classifiers(X_iris, y_iris, "Iris (Clean)")
evaluate_classifiers(X_iris_noisy, y_iris, "Iris (Noisy)")
