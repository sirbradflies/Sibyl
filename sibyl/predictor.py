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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, make_scorer

from sibyl.encoders.omniencoder import OmniEncoder
from sibyl.models.kerasmodels import KerasDenseRegressor, KerasDenseClassifier

PARAMS = {"pca__n_components": [None, 0.99, 0.90],
          "model__units": [(64,), (64, 64), (64, 64, 64)],
          "model__batch_norm": [True, False]}


class SibylBase(Pipeline):
    def search(self, X, y, params=PARAMS, groups=None, refit=True,
               scoring=None, cv=None, n_jobs=-1):
        """
        Random search for the best model and return the search results

        :param X : Dataframe or Array
        Features for training
        :param Y : Series or Array
        Target for training
        :param params : Dictionary or list of Dictionaries
        Default parameters for standard pipeline are:
        >>>{"pca__n_components": [None, 0.99, 0.90],
        >>> "model__units": [(64,), (64, 64), (64, 64, 64)],
        >>> "model__batch_norm": [True, False]}
        Search parameters to be used with the pipeline as in SKLearn GridSearchCV.
        Required if the pipeline steps are customized.
        :param groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into train/test set.
        : param refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole dataset.
        :param scoring : gstr, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on the test set.
        :param cv: int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        :param n_jobs : int, default=-1
        Number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
        :return Dataframe with search results
        """
        search = GridSearchCV(self, params, refit=refit, scoring=scoring,
                              verbose=2, cv=cv, n_jobs=n_jobs)
        search.fit(X, y, groups=groups)
        self.set_params(**search.best_params_).fit(X, y)  # TODO: Double fit! Remove
        results = pd.DataFrame(search.cv_results_)
        return results

    def score(self, X, y):
        """ Score features X against target y """
        return self.scorer(self, X, y)

    def __str__(self):
        steps = [type(obj).__name__ for _,obj in self.get_params()["steps"]]
        return "Sibyl_"+"_".join(steps)

    def save(self, file):
        """
        Save the complete predictor pipeline

        :param file : ByteStream or string
        Save the file in the bytestream or path passed
        """
        if type(file) == str:
            with open(file, "wb") as f:
                joblib.dump(self, f)
        else:
            joblib.dump(self, file)


def load(file):
    """
    Load the complete predictor pipeline

    :param file : ByteStream or string
    Load the file from the bytestream or path passed
    :return : Predictor pipeline with the saved status
    """
    if type(file) == str:
        with open(file, "rb") as f:
            return joblib.load(f)
    else:
        return joblib.load(file)


class SibylClassifier(SibylBase):
    """
        Set up the SybilClassifier with the desired steps

        :param steps: List of tuples, default None. If None it defaults to the following steps:
        [("omni", OmniEncoder()), ("pca", PCA()), ("model", KerasDenseClassifier())].
        Same as Pipeline, accepts a list of tuples ("STEP_NAME", "ESTIMATOR")
        :param scorer: SKLearn compatible scorer, default None. If None defaults to accuracy score.

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
    def __init__(self, steps=None):
        if steps is None:
            steps = [("omni", OmniEncoder()),
                     ("pca", PCA()),
                     ("model", KerasDenseClassifier())]
        super(SibylClassifier, self).__init__(steps=steps)


class SibylRegressor(SibylBase):
    """
            Set up the SybilClassifier with the desired steps

            :param steps: List of tuples, default None. If None it defaults to the following steps:
            [("omni", OmniEncoder()), ("pca", PCA()), ("model", KerasDenseRegressor())].
            Same as Pipeline, accepts a list of tuples ("STEP_NAME", "ESTIMATOR")
            :param scorer: SKLearn compatible scorer, default None. If None defaults to R2 score.

        Examples
        --------
        >>> from sklearn.svm import SVR
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.pipeline import Pipeline
        >>> X, y = make_regression(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
        ...                                                     random_state=0)
        >>> pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
        >>> pipe.fit(X_train, y_train)
        Pipeline(steps=[('scaler', StandardScaler()), ('svr', SVR())])
        >>> pipe.score(X_test, y_test)
        0.88
        """
    def __init__(self, steps=None):
        if steps is None:
            steps = [("omni", OmniEncoder()),
                     ("pca", PCA()),
                     ("model", KerasDenseRegressor())]
        super(SibylRegressor, self).__init__(steps=steps)
