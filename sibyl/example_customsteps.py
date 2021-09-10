"""
MNIST Classifier with custom steps

@author: Francesco Baldisserri
@creation date: 13/07/2021
"""

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sibyl import predictor as pred
from sibyl.models import kerasdense as kd

STEPS = [("preprocessing", PCA()),
         ("model", kd.KerasDenseClassifier(n_iter_no_change=1,
                                           val_split=0.2,
                                           epochs=10))]

SEARCH_PARAMS = {"preprocessing__n_components": [None, 0.99, 0.90],
                 "model__units": [(64,), (64, 64), (64, 64, 64)],
                 "model__batch_norm": [True, False]}


def main():
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    predclf = pred.SibylClassifier(steps=STEPS)
    predclf.search(X_train, y_train, params=SEARCH_PARAMS)  # RandomizedGridSearchCV available but also fit method is available
    predclf.save("example_model.sibyl")
    predclf = pred.load("example_model.sibyl")
    print(f"Test score: {predclf.score(X_test, y_test):.4f}")


if __name__ == '__main__':
    main()
