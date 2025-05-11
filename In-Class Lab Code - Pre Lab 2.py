import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
cm_bright = ListedColormap(["#FF0000","#0000FF"])

def setSubPlot(ax, X, title, featureNameX, featureNameY):
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.set_xlabel(featureNameX)
    ax.set_ylabel(featureNameY)
    

classifiers = {
    "KNN (k=5)":              KNeighborsClassifier(5),
    "Decision Trees":   DecisionTreeClassifier(max_depth=5),
    "Random Forests":   RandomForestClassifier(max_depth=5, n_estimators=10),
    "Naive Bayes":      GaussianNB(),
    "Linear SVM":       SVC(kernel="linear", C=0.025),
    "Kernel SVM":       SVC(gamma=2, C=1),
    "Neural Networks":  MLPClassifier(alpha=0.1, max_iter=1000)
    }


scaler = StandardScaler()

dataset = {
    "moons (original Data)": make_moons(noise=0.05, random_state=0),
    "circles (original Data)": make_circles(noise=0.05,random_state=0),
    "blobs": make_blobs(random_state=100, centers=2, cluster_std=4.0)
}

figure = plt.figure(figsize=(27, 9))

i = 1
for datasetname, dataset in dataset.items():
    X,y = dataset
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=42)
    # Original Data
    ax = plt.subplot(3, 8, i)

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    setSubPlot(ax, X, datasetname, "X0", "X1")
    # Do it for all classifiers
    for classifiername, classifier in classifiers.items():
        clf = classifier
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(Xtrain,ytrain)
        score = clf.score(Xtest,ytest)
        ax = plt.subplot(3, 8, i+1)

        ax.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain, cmap=cm_bright,edgecolor="k")
        ax.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest, cmap=cm_bright,edgecolor="r",alpha=0.5)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=plt.cm.RdBu, ax=ax, alpha=0.2, eps=0.5)
        setSubPlot(ax,X,classifiername+f": ({score*100:.1f})","X0","X1")
        i+=1
    i+=1

plt.tight_layout()
plt.show()
