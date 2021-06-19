"""
MNIST Classifier using Sibyl

@author: Francesco Baldisserri
@creation date: 24/07/2020
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sibyl import predictor as pred


def main():
    X, y = make_classification(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    predclf = pred.SibylClassifier()
    predclf.search(X_train, y_train)
    print(f"Score on MNIST dataset: {predclf.score(X_test, y_test):.4f}")


if __name__ == '__main__':
    main()
