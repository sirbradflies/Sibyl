"""
Light GBM wrapper to simplify early stopping

@author: Francesco Baldisserri
@creation date: 24/9/2021
"""

from math import log10
import lightgbm as lgb
from sklearn.model_selection import train_test_split


class LGBMRegressorWrapper(lgb.LGBMRegressor):
    def fit(self, x, y):
        rounds = int(log10(self.n_estimators))
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        callbacks = [lgb.early_stopping(rounds), lgb.log_evaluation(period=0)]
        return super().fit(x_train, y_train, eval_set=[(x_val, y_val)],
                           callbacks=callbacks)


class LGBMClassifierWrapper(lgb.LGBMClassifier):
    def fit(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        rounds = int(log10(self.n_estimators))
        callbacks = [lgb.early_stopping(rounds), lgb.log_evaluation(period=0)]
        return super().fit(x_train, y_train, eval_set=[(x_val, y_val)],
                           callbacks=callbacks)