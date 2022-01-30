"""
Basic example of regressor using Sibyl

@author: Francesco Baldisserri
@creation date: 24/07/2020
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sibyl import predictor as pred


def main():
    X, y = datasets.fetch_california_housing(return_X_y=True)  # No encoding needed since OmniEncoder recognizes discrete, continuous and categoricals
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    predrgr = pred.SibylRegressor()
    predrgr.search(X_train, y_train)  # RandomizedGridSearchCV available but also fit method is available
    print(f"Test score: {predrgr.score(X_test, y_test):.4f}")


if __name__ == '__main__':
    main()
