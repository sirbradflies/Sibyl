"""
Sklearn Pipeline wrapper that simplifies ML flow and
works as a simple AutoML tool.

@author: Francesco Baldisserri
@creation date: 20/02/2020
@version: 1.0
"""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score, make_scorer

from sibyl.encoders.omniencoder import OmniEncoder
from sibyl.models.kerasdense import KerasDenseRegressor, KerasDenseClassifier

PARAMS = {"pca__n_components": [None, 0.99, 0.90],
          "model__units": [(64,), (64, 64), (64, 64, 64)],
          "model__batch_norm": [True, False]}


class SibylBase(Pipeline):
    """
    Simple AutoML class to solve basic ML tasks.

    Attributes
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform)
        with the last object an estimator.
    scorer : function or a dict
        Scorer for model cross validation.
    """
    def __init__(self, steps, scorer):
        self.scorer = scorer
        super(SibylBase, self).__init__(steps)

    def search(self, X, y, params=PARAMS, groups=None,
               cv=None, n_iter=10, n_jobs=-1):
        """
        Randomized search for the best model and return the best model score.

        :param X: array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        :param y: array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        :param params: dict of str -> object, default = standard Keras params
            Parameters passed to the ``fit`` method of the estimator.
        :param groups: array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        :param cv: int, default=None
        Cross-validation generator or an iterable.
        :param n_iter: int, default=10
        Number of search iterations to perform.
        :param n_jobs: int, default=None
        Number of jobs to run in parallel.
        :return: float with best score found during the search
        """
        search = RandomizedSearchCV(self, params, scoring=self.scorer,
                                    refit=False, verbose=5, cv=cv,
                                    n_iter=n_iter, n_jobs=n_jobs)
        search.fit(X, y, groups=groups)
        self.set_params(**search.best_params_).fit(X, y)
        results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        print(results[["params", "mean_test_score",
                       "std_test_score", "mean_fit_time"]].to_string())
        return search.best_score_

    def score(self, X, y):
        """ Score features X against target y """
        return self.scorer(self, X, y)

    def __str__(self):
        steps = [type(obj).__name__ for _,obj in self.get_params()["steps"]]
        return "Sibyl_"+"_".join(steps)

    def save(self, file):
        """
        Save predictor pipeline to a file
        :param file: file name or IO object
        """
        if type(file) == str:
            with open(file, "wb") as f:
                joblib.dump(self, f)
        else:
            joblib.dump(self, file)


def load(file):
    """
    Load predictor pipeline from a file
    :param file: file name or IO object
    """
    if type(file) == str:
        with open(file, "rb") as f:
            return joblib.load(f)
    else:
        return joblib.load(file)


class SibylClassifier(SibylBase):
    """
    Simple AutoML classifier to solve basic ML tasks.

    Attributes
    ----------
    steps : list, default = OmniEncoder, PCA, KerasDenseClassifier
        List of (name, transform) tuples (implementing fit/transform)
        with the last object an estimator.
    scorer : function or a dict, default = accuracy score
        Scorer for model cross validation.
    """
    def __init__(self, steps=None, scorer=None):
        if steps is None:
            steps = [("omni", OmniEncoder()),
                     ("pca", PCA()),
                     ("model", KerasDenseClassifier())]
        if scorer is None:
            scorer = make_scorer(accuracy_score)
        super(SibylClassifier, self).__init__(steps=steps, scorer=scorer)


class SibylRegressor(SibylBase):
    """
    Simple AutoML regressor to solve basic ML tasks.

    Attributes
    ----------
    steps : list, default = OmniEncoder, PCA, KerasDenseRegressor
        List of (name, transform) tuples (implementing fit/transform)
        with the last object an estimator.
    scorer : function or a dict, default = accuracy score
        Scorer for model cross validation.
    """
    def __init__(self, steps=None, scorer=None):
        if steps is None:
            steps = [("omni", OmniEncoder()),
                     ("pca", PCA()),
                     ("model", KerasDenseRegressor())]
        if scorer is None:
            scorer = make_scorer(r2_score)
        super(SibylRegressor, self).__init__(steps=steps, scorer=scorer)
