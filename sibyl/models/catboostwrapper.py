"""
CatBoost wrapper to simplify early stopping

@author: Francesco Baldisserri
@creation date: 21/02/2020
"""

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split


class CBoostRegressorWrapper(CatBoostRegressor):
    def fit(self, x, y):
        if "early_stopping_rounds" in self.get_params():
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
            return super().fit(x_train, y_train, eval_set=(x_val, y_val))
        else:
            return super().fit(x, y)


class CBoostClassifierWrapper(CatBoostClassifier):
    def fit(self, x, y):
        if "early_stopping_rounds" in self.get_params():
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
            return super().fit(x_train, y_train, eval_set=(x_val, y_val))
        else:
            return super().fit(x, y)

"""class CBoostRegressorWrapper_TOFIX(RegressorMixin, BaseEstimator, CatBoostRegressor):
    def __init__(self, validation_fraction=0, *args, **kwargs):  # TODO: Fix issue with parameter validation_fraction to setup from init
        self.validation_fraction = validation_fraction
        super(CBoostRegressorWrapper_TOFIX, self).__init__(*args, **kwargs)
        print(f"Deep Params: {self.get_params(deep=True)}")
        print(f"Shallow Params: {self.get_params(deep=False)}")

    def fit(self, x, y):
        if self.validation_fraction > 0:
            x_train, x_val, y_train, y_val = train_test_split(x, y)
            return super().fit(x_train, y_train, eval_set=(x_val, y_val))
        else:
            return super().fit(x, y)"""
