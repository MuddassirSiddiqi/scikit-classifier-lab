import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Prepare synthetic datasets
synthetic_datasets = {
    "Moons": make_moons(noise=0.3, random_state=0),
    "Circles": make_circles(noise=0.2, factor=0.5, random_state=1),
    "Blobs": make_blobs(n_samples=300, centers=2, cluster_std=1.5, random_state=2)
}

# Define hyperparameter grids
param_grids = {
    "KNN": {"kneighborsclassifier__n_neighbors": [3, 5, 7, 9]},
    "Decision Tree": {"decisiontreeclassifier__max_depth": [3, 5, 7, 9]},
    "Random Forest": {
        "randomforestclassifier__max_depth": [3, 5, 7],
        "randomforestclassifier__n_estimators": [10, 50, 100]
    },
    "Linear SVM": {"svc__C": [0.01, 0.025, 0.1, 1]},
    "Kernel SVM": {"svc__gamma": [0.5, 1, 2, 4]}
}

# Classifiers with default parameters for tuning
tuned_classifiers = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Decision Tree": make_pipeline(StandardScaler(), DecisionTreeClassifier()),
    "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier()),
    "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear")),
    "Kernel SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf"))
}

# Evaluate and tune
def tune_and_evaluate(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    for name, pipeline in tuned_classifiers.items():
        grid = GridSearchCV(pipeline, param_grids[name], cv=3)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{dataset_name} - {name} Accuracy: {acc:.4f}, Best Params: {grid.best_params_}")

# Apply tuning to each dataset
for dataset_name, (X, y) in synthetic_datasets.items():
    tune_and_evaluate(X, y, dataset_name)
