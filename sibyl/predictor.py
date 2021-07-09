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

    :param steps : list
    List of (name, transform) tuples (implementing fit/transform)
    with the last object an estimator.
    :param scorer : str or Scorer
    Scorer for model cross validation.
    """
    def __init__(self, steps, scorer):
        self.scorer = scorer
        super(SibylBase, self).__init__(steps)

    def search(self, X, y, params=PARAMS, groups=None,
               cv=None, n_iter=10, n_jobs=-1):
        """
        Random search for the best model and return the search results

        :param X : Dataframe or Array
        Features for training
        :param Y : Series or Array
        Target for training
        :param groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into train/test set.
        :param cv : int, cross-validation generator or an iterable, default=None
        Cross-validation splitting strategy (see SciKit learn docs for more details)
        :param n_iter : int, default=10
        Number of parameter settings that are sampled (see SciKit learn docs for more details).
        :param n_jobs : int, default=-1
        Number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
        :return Dataframe with search results
        """
        search = RandomizedSearchCV(self, params, scoring=self.scorer,
                                    refit=False, verbose=5, cv=cv,
                                    n_iter=n_iter, n_jobs=n_jobs)
        search.fit(X, y, groups=groups)
        self.set_params(**search.best_params_).fit(X, y)
        results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        return results[["params", "mean_test_score",
                        "std_test_score", "mean_fit_time"]]

    def score(self, X, y):
        """ Score features X against target y """
        return self.scorer(self, X, y)

    def __str__(self):
        steps = [type(obj).__name__ for _,obj in self.get_params()["steps"]]
        return "Sibyl_"+"_".join(steps)

    def save(self, file):
        if type(file) == str:
            with open(file, "wb") as f:
                joblib.dump(self, f)
        else:
            joblib.dump(self, file)


def load(file):
    if type(file) == str:
        with open(file, "rb") as f:
            return joblib.load(f)
    else:
        return joblib.load(file)


class SibylClassifier(SibylBase):
    """
        Set up the SybilClassifier with the desired steps

        :param steps: List of tuples, default None. If None it defaults to the following steps:
        [("omni", OmniEncoder()), ("pca", PCA()), ("catboost", CatBoostClassifier(logging_level='Silent'))].
        Same as Pipeline, accepts a list of tuples ("STEP_NAME", "ESTIMATOR")
        :param scorer: SKLearn compatible scorer, default None. If None defaults to F1 score.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    >>> pipe.fit(X_train, y_train)
    Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
    >>> pipe.score(X_test, y_test)
    0.88
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
    def __init__(self, steps=None, scorer=None):
        if steps is None:
            steps = [("omni", OmniEncoder()),
                     ("pca", PCA()),
                     ("model", KerasDenseRegressor())]
        if scorer is None:
            scorer = make_scorer(r2_score)
        super(SibylRegressor, self).__init__(steps=steps, scorer=scorer)
