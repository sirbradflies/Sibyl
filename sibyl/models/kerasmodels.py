"""
Generic regression model for machine learning problems based on Keras

@author: Francesco Baldisserri
@creation date: 06/03/2020
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import scipy
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, \
    Dropout, Conv1D, Reshape


# TODO: Refactor with Estimator template:
# https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
class KerasDenseRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, units=(64, 64), dropout=0, activation="relu",
                 batch_norm=False, batch_size=None, optimizer="nadam",
                 kernel_init="glorot_normal", val_split=0, epochs=10,
                 loss="mean_squared_error", out_act="linear",
                 n_iter_no_change=None, custom_objects=None,
                 _estimator_type="regressor"):
        self.units = units
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.kernel_init = kernel_init
        self.val_split = val_split
        self.epochs = epochs
        self.loss = loss
        self.out_act = out_act
        self.n_iter_no_change = n_iter_no_change
        self.model = None
        self.custom_objects = custom_objects
        self._estimator_type = _estimator_type

    def fit(self, X, y, metrics=None):
        X_val, y_val = _validate_array(X), _validate_array(y)
        self._build_model(X_val, y_val, metrics)
        calls = []
        if self.n_iter_no_change is not None:
            early_stop_metric = "val_loss" if self.val_split > 0 else "loss"
            calls += [EarlyStopping(monitor=early_stop_metric,
                                    patience=self.n_iter_no_change,
                                    restore_best_weights=True)]
        return self.model.fit(X_val, y_val, epochs=self.epochs, verbose=0,
                              callbacks=calls, validation_split=self.val_split)

    def predict(self, X):
        return self.model.predict(_validate_array(X))

    def _build_model(self, X, y, metrics):
        """ Build Keras model according to input parameters """
        layers = [Flatten()] if len(X.shape[1:]) > 1 else []
        for units in self.units:
            if self.dropout > 0: layers += [Dropout(self.dropout)]
            layers += [Dense(units,
                             activation=self.activation,
                             kernel_initializer=self.kernel_init)]
            if self.batch_norm: layers += [BatchNormalization()]
        out_shape = y.shape[1] if len(y.shape) > 1 else 1
        layers += [Dense(out_shape, activation=self.out_act)]
        self.model = keras.models.Sequential(layers)
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=metrics)


class KerasDenseClassifier(KerasDenseRegressor, ClassifierMixin):
    def __init__(self, units=(64, 64), dropout=0, activation="relu",
                 batch_norm=False, batch_size=None, optimizer="nadam",
                 kernel_init="glorot_normal", val_split=0, epochs=10,
                 loss="categorical_crossentropy", out_act="softmax",
                 n_iter_no_change=None, custom_objects=None,
                 _estimator_type="classifier"):
        self.encoder = OneHotEncoder(drop="first")
        super().__init__(units=units, dropout=dropout,
                         activation=activation, batch_norm=batch_norm,
                         batch_size=batch_size,  optimizer=optimizer,
                         kernel_init=kernel_init, val_split=val_split,
                         epochs=epochs, loss=loss, out_act=out_act,
                         n_iter_no_change=n_iter_no_change,
                         custom_objects=custom_objects)

    def fit(self, X, y):  # Suboptimally using Softmax also for 2-class problems
        if y.ndim == 1:
            y = np.array(y).reshape(-1, 1)
        y_enc = self.encoder.fit_transform(y)
        return super().fit(X, y_enc, ["accuracy"])

    def predict(self, X):
        y_enc = super().predict(X)
        return self.encoder.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class KerasCNNRegressor(KerasDenseRegressor):
    def __init__(self, cnn_units=(64, 64), dense_units=(64,64), dropout=0,
                 activation="relu", batch_norm=False, batch_size=None,
                 optimizer="nadam", kernel_init="glorot_normal", val_split=0,
                 epochs=10, loss="mean_squared_error", out_act="linear",
                 n_iter_no_change=None, custom_objects=None,
                 _estimator_type="regressor"):
        self.cnn_units = cnn_units
        self.dense_units = dense_units
        super().__init__(dropout = dropout,
                         activation = activation,
                         batch_norm = batch_norm,
                         batch_size = batch_size,
                         optimizer = optimizer,
                         kernel_init = kernel_init,
                         val_split = val_split,
                         epochs = epochs,
                         loss = loss,
                         out_act = out_act,
                         n_iter_no_change = n_iter_no_change,
                         custom_objects = custom_objects,
                         _estimator_type = _estimator_type)

    def _build_model(self, X, y, metrics):
        """ Build Keras model according to input parameters """
        layers = [Reshape(X.shape[1:]+(1,))] if len(X.shape[1:]) < 2 else []
        for units in self.cnn_units:
            if self.dropout > 0: layers += [Dropout(self.dropout)]
            layers += [Conv1D(filters=units, kernel_size=2, padding="causal",
                              activation=self.activation, dilation_rate=2),]
            if self.batch_norm: layers += [BatchNormalization()]
        layers += [Flatten()]
        for units in self.dense_units:
            if self.dropout > 0: layers += [Dropout(self.dropout)]
            layers += [Dense(units,
                             activation=self.activation,
                             kernel_initializer=self.kernel_init)]
            if self.batch_norm: layers += [BatchNormalization()]
        out_shape = y.shape[1] if len(y.shape) > 1 else 1
        layers += [Dense(out_shape, activation=self.out_act)]
        self.model = keras.models.Sequential(layers)
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=metrics)


class KerasCNNClassifier(KerasCNNRegressor):
    def __init__(self, cnn_units=(64, 64), dense_units=(64, 64), dropout=0,
                 activation="relu", batch_norm=False, batch_size=None,
                 optimizer="nadam", kernel_init="glorot_normal", val_split=0,
                 epochs=10, loss="categorical_crossentropy", out_act="softmax",
                 n_iter_no_change=None, custom_objects=None,
                 _estimator_type="classifier"):
        self.encoder = OneHotEncoder(drop="first")
        super().__init__(
            cnn_units = cnn_units,
            dense_units= dense_units,
            dropout = dropout,
            activation = activation,
            batch_norm = batch_norm,
            batch_size = batch_size,
            optimizer = optimizer,
            kernel_init = kernel_init,
            val_split = val_split,
            epochs = epochs,
            loss = loss,
            out_act = out_act,
            n_iter_no_change = n_iter_no_change,
            custom_objects = custom_objects,
            _estimator_type = _estimator_type)

    def fit(self, X, y):  # Suboptimally using Softmax also for 2-class problems
        if y.ndim == 1:
            y = np.array(y).reshape(-1, 1)
        y_enc = self.encoder.fit_transform(y)
        return super().fit(X, y_enc, ["accuracy"])

    def predict(self, X):
        y_enc = super().predict(X)
        return self.encoder.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _validate_array(array):
    if isinstance(array, list):
        validated_array = np.array(array)
    elif scipy.sparse.issparse(array):
        validated_array = array.todense()
    else:
        validated_array = array
    return validated_array
