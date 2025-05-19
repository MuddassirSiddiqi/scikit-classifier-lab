

```markdown
# Scikit-Classifier-Lab

This project explores and compares the performance of several classification algorithms using both real and synthetic datasets. It includes baseline accuracy evaluations, noise sensitivity tests, and hyperparameter tuning using `GridSearchCV`.

## 📁 Repository Structure

This repository contains two core parts:

- **Part A**: Evaluation of various classifiers on the Iris dataset (clean and noisy versions).
- **Part B**: Hyperparameter tuning and evaluation on synthetic datasets (moons, circles, and blobs).

---

## 🧠 Algorithms Used

The following classifiers are implemented and compared:

- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes (GaussianNB)
- Support Vector Machine (Linear and Kernel)

---

## 📌 Part A – Classification on Iris Dataset

### Description

This script loads the Iris dataset, adds Gaussian noise to evaluate classifier robustness, and tests multiple classifiers using standard scaling and accuracy as the performance metric.

### Features

- Evaluation of classifiers on clean and noisy datasets.
- Pipeline integration with `StandardScaler`.
- Accuracy comparison printed for both dataset variants.

### Example Output

```

Iris (Clean) - Linear SVM Accuracy: 0.9778
Iris (Noisy) - Linear SVM Accuracy: 0.9333
...

```

---
![Output Image](https://github.com/MuddassirSiddiqi/scikit-classifier-lab/blob/master/PartA%20Output.png)
## 🔍 Part B – Hyperparameter Tuning on Synthetic Datasets

### Description

This part uses synthetic datasets (`make_moons`, `make_circles`, `make_blobs`) to test classifier performance under different geometrical data distributions. It applies `GridSearchCV` for hyperparameter tuning.

### Features

- Grid search for optimal model parameters.
- Pipelines with scaling and classifier integration.
- Outputs best hyperparameters and final accuracy scores.

### Example Output

```

Moons - Random Forest Accuracy: 0.9000, Best Params: {'randomforestclassifier\_\_max\_depth': 5, 'randomforestclassifier\_\_n\_estimators': 50}

````

---
![Output Image](https://github.com/MuddassirSiddiqi/scikit-classifier-lab/blob/master/PartB%20Output.png)
## 🧪 Installation and Usage

### Requirements

- Python 3.x
- scikit-learn
- numpy

### Run Scripts

```bash
# Clone the repository
git clone https://github.com/MuddassirSiddiqi/scikit-classifier-lab.git
cd scikit-classifier-lab

# Run Part A
python part_a.py

# Run Part B
python part_b.py
````

---

## 🗃️ Dataset Sources

* **Iris Dataset**: Provided by `sklearn.datasets.load_iris`
* **Synthetic Datasets**: Generated using `make_moons`, `make_circles`, and `make_blobs`

---

## 📈 Future Enhancements

* Visualization of decision boundaries.
* Inclusion of ensemble techniques like stacking or boosting.
* Evaluation using additional metrics (F1, ROC AUC).

---

## 📜 License

This project is released under the [MIT License](LICENSE).

```


```
